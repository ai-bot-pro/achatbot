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
        def __init__(self, input_size, hidden_size, num_experts, rank, world_size):
            super(MoE, self).__init__()
            self.rank = rank
            self.world_size = world_size
            self.num_experts = num_experts

            # 门控网络在所有设备上运行
            self.gating = GatingNetwork(input_size, num_experts)

            # 每个设备只拥有一个专家
            self.expert = Expert(input_size, hidden_size).to(f"cuda:{rank}")

        def forward(self, x):
            # 门控网络计算每个专家的得分
            gate_scores = self.gating(x)  # [batch_size, num_experts]

            # 选择得分最高的专家（top-1）
            top_expert_idx = torch.argmax(gate_scores, dim=-1)  # [batch_size]

            # 当前设备的专家 ID
            local_expert_id = self.rank

            # 初始化输出
            batch_size = x.size(0)
            output = torch.zeros(batch_size, self.expert.fc.out_features).to(x.device)

            # 处理分配给当前设备的输入
            mask = top_expert_idx == local_expert_id
            if mask.any():
                local_input = x[mask]
                local_output = self.expert(local_input)
                output[mask] = local_output

            # 使用 All-Gather 收集所有专家的输出
            global_output = [torch.zeros_like(output) for _ in range(self.world_size)]
            dist.all_gather(global_output, output)

            # 根据门控得分加权合并输出
            final_output = torch.zeros_like(output)
            for i in range(self.world_size):
                final_output += global_output[i] * gate_scores[:, i].unsqueeze(-1)

            return final_output

    # 专家并行训练函数
    def train_expert_parallel(rank, world_size, input_size, hidden_size, data, labels):
        # 设置分布式环境
        setup_distributed(rank, world_size)

        # 参数设置
        num_experts = world_size  # 每个 GPU 一个专家
        device = torch.device(f"cuda:{rank}")

        # 将数据移动到设备
        data, labels = data.to(device), labels.to(device)

        # 初始化 MoE 模型
        model = MoE(input_size, hidden_size, num_experts, rank, world_size).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()  # 假设回归任务
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
    data = torch.randn(batch_size, input_size)
    labels = torch.randn(batch_size, hidden_size)  # 假设回归任务

    # 启动进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=train_expert_parallel,
            args=(rank, world_size, input_size, hidden_size, data, labels),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
