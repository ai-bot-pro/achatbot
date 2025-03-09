import os
import modal


app = modal.App("train-demo-ep")

demo_image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


@app.function(
    image=demo_image,
    gpu=os.getenv("IMAGE_GPU", "T4:2"),
    retries=0,
    # how long should we stay up with no requests?
    container_idle_timeout=15 * 60,
)
def run():
    import torch

    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    if world_size < 2:
        print("需要至少 2 个 GPU 来演示分布式数据并行")
        return

    print(f"Running Expert Parallelism with {world_size} GPUs")

    train_ep(world_size)


# modal run src/train/demo/ep.py
@app.local_entrypoint()
def main():
    run.remote()


# 训练函数
def train_ep(world_size):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    import os
    from multiprocessing import Process

    # 初始化分布式环境
    def setup_distributed(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # 清理分布式环境
    def cleanup():
        dist.destroy_process_group()

    # 定义单个专家（一个简单的线性层）
    class Expert(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Expert, self).__init__()
            self.fc = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.fc(x))

    # 定义门控网络
    class GatingNetwork(nn.Module):
        def __init__(self, input_size, num_experts):
            super(GatingNetwork, self).__init__()
            self.fc = nn.Linear(input_size, num_experts)  # 输出每个专家的得分

        def forward(self, x):
            return torch.softmax(self.fc(x), dim=-1)  # 返回每个专家的概率

    # MoE 模型
    class MoE(nn.Module):
        def __init__(self, input_size, hidden_size, num_experts, rank, world_size, top_k):
            super(MoE, self).__init__()
            self.rank = rank
            self.world_size = world_size
            self.num_experts = num_experts
            self.top_k = top_k

            # 门控网络在所有设备上运行
            self.gating = GatingNetwork(input_size, num_experts)

            # 每个设备只拥有一个专家
            self.expert = Expert(input_size, hidden_size).to(f"cuda:{rank}")

        def forward(self, x):
            # 门控网络计算每个专家的得分
            gate_scores = self.gating(x)  # [batch_size, num_experts]

            # 选择 Top-k 专家
            topk_scores, topk_indices = torch.topk(
                gate_scores, self.top_k, dim=-1
            )  # [batch_size, k]

            # 当前设备的专家 ID
            local_expert_id = self.rank

            # 初始化输出为具有梯度的张量
            batch_size = x.size(0)
            outputs = []  # 使用列表存储每个样本的输出
            for i in range(batch_size):
                sample_output = torch.zeros(
                    self.expert.fc.out_features, device=x.device, requires_grad=True
                )
                if local_expert_id in topk_indices[i]:
                    local_output = self.expert(x[i : i + 1])  # 处理单个样本
                    idx = (topk_indices[i] == local_expert_id).nonzero(as_tuple=True)[0]
                    score = topk_scores[i, idx]
                    sample_output = local_output.squeeze(0) * score  # 加权输出
                outputs.append(sample_output)

            # 将输出堆叠成批次
            output = torch.stack(outputs)

            # 创建用于收集输出的列表
            global_output = []
            # 为每个进程创建一个副本
            for _ in range(self.world_size):
                global_output.append(torch.zeros_like(output))

            # 使用 broadcast 来收集所有专家的输出
            for i in range(self.world_size):
                if self.rank == i:
                    # 当前进程广播其输出
                    broadcast_tensor = output
                else:
                    # 其他进程接收广播
                    broadcast_tensor = torch.zeros_like(output)
                # 广播操作
                dist.broadcast(broadcast_tensor, src=i)
                global_output[i] = broadcast_tensor

            # 合并所有专家的加权输出
            final_output = sum(global_output)  # 直接累加（已加权）

            return final_output

    # 专家并行训练函数
    def train_expert_parallel(rank, world_size, input_size, hidden_size, data, labels, top_k):
        # 设置分布式环境
        setup_distributed(rank, world_size)

        # 参数设置
        num_experts = world_size  # 每个 GPU 一个专家
        device = torch.device(f"cuda:{rank}")

        # 将数据移动到设备
        data, labels = data.to(device), labels.to(device)

        # 初始化 MoE 模型
        model = MoE(input_size, hidden_size, num_experts, rank, world_size, top_k).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()  # 假设回归任务
        # criterion = nn.CrossEntropyLoss()  # 分类任务
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 训练步骤
        model.train()
        for epoch in range(5):  # 模拟 5 个 epoch
            optimizer.zero_grad()

            # 前向传播
            output = model(data)
            loss = criterion(output, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 打印损失
            dist.barrier()  # 同步所有进程
            if rank == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        cleanup()

    # 生成模拟数据
    batch_size = 8
    input_size = 16
    hidden_size = 32
    top_k = 2
    data = torch.randn(batch_size, input_size, requires_grad=True)
    labels = torch.randn(batch_size, hidden_size, requires_grad=True)  # 启用梯度

    # 启动进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=train_expert_parallel,
            args=(rank, world_size, input_size, hidden_size, data, labels, top_k),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
