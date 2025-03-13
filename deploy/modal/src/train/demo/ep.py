# https://pytorch.org/blog/training-moes/
import os
import modal


app = modal.App("train-demo-ep")

demo_image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


@app.function(
    image=demo_image,
    gpu=os.getenv("IMAGE_GPU", "T4:2"),
    retries=0,
    # how long should we stay up with no requests?
    scaledown_window=15 * 60,
)
def run(num_experts=4, top_k=2, alpha=0.01, mode="brodcast"):
    import torch

    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    if world_size < 2:
        print("需要至少 2 个 GPU 来演示分布式数据并行")
        return
    if num_experts < top_k:
        print(f"top_k {top_k} 必须 小于 总专家数 {num_experts}")
        return

    print(
        f"Running Expert Parallelism mode {mode} with {world_size} GPUs, Experts/GPU: {num_experts//world_size}, top_k: {top_k}, load balance loss alpha: {alpha}"
    )

    train_ep(world_size, num_experts, top_k, alpha, mode)


"""
# defualt mode=brodcast num-experts=2, top-k=1, alpha=0.0 on 2 GPU
modal run src/train/demo/ep.py
modal run src/train/demo/ep.py --num-experts 2 --top-k 2 --alpha 0.1
modal run src/train/demo/ep.py --num-experts 2 --top-k 1 --alpha 0.1
modal run src/train/demo/ep.py --num-experts 4 --top-k 2 --alpha 0.01

modal run src/train/demo/ep.py --mode all2all
modal run src/train/demo/ep.py --mode all2all --num-experts 2 --top-k 2 --alpha 0.1
modal run src/train/demo/ep.py --mode all2all --num-experts 2 --top-k 1 --alpha 0.1
modal run src/train/demo/ep.py --mode all2all --num-experts 4 --top-k 2 --alpha 0.01
"""


@app.local_entrypoint()
def main(num_experts: str = "2", top_k: str = "1", alpha: str = "0.0", mode="brodcast"):
    run.remote(int(num_experts), int(top_k), float(alpha), mode)


# 训练函数
def train_ep(world_size, num_experts=4, top_k=2, alpha=0.01, mode="brodcast"):
    import os
    from multiprocessing import Process
    import torch
    import torch.nn as nn
    import torch.distributed as dist

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
        def __init__(
            self, input_size, hidden_size, num_experts, rank, world_size, top_k, alpha=0.01
        ):
            super(MoE, self).__init__()
            self.rank = rank
            self.world_size = world_size
            self.num_experts = num_experts
            self.top_k = top_k
            self.alpha = alpha  # 负载均衡损失的权重
            self.experts_per_gpu = num_experts // world_size
            self.device = torch.device(f"cuda:{rank}")

            # 门控网络在所有设备上运行
            self.gating = GatingNetwork(input_size, num_experts)

            # 每个 GPU 分配部分专家
            self.local_experts = nn.ModuleList(
                [Expert(input_size, hidden_size) for _ in range(self.experts_per_gpu)]
            ).to(self.device)

            # 本地专家的全局索引范围
            self.start_expert_idx = rank * self.experts_per_gpu
            self.end_expert_idx = (rank + 1) * self.experts_per_gpu
            print(f"rank {rank} expert_idx range [{self.start_expert_idx} {self.end_expert_idx})")

        def compute_load_balance_loss(self, gate_scores, topk_indices):
            # 计算每个专家的使用频率（fraction of tokens）
            batch_size = gate_scores.size(0)
            expert_usage = torch.zeros(self.num_experts, device=gate_scores.device)
            for i in range(batch_size):
                for idx in topk_indices[i]:
                    expert_usage[idx] += 1
            f = expert_usage / (batch_size * self.top_k)  # 归一化使用频率

            # 计算门控得分的平均值（gate probability）
            p = gate_scores.mean(dim=0)  # [num_experts]

            # 负载均衡损失：f 和 p 的乘积方差
            load_loss = torch.mean(f * p) ** 2 + torch.var(f * p)
            return self.alpha * load_loss

        def forward(self, x):
            # 门控网络计算每个专家的得分
            gate_scores = self.gating(x)  # [batch_size, num_experts]

            # 选择 Top-k 专家
            topk_scores, topk_indices = torch.topk(
                gate_scores, self.top_k, dim=-1
            )  # [batch_size, k]

            # 计算负载均衡损失
            load_balance_loss = self.compute_load_balance_loss(gate_scores, topk_indices)
            if self.rank == 0:
                pass
                # print once
                # print(f"topk_scores {topk_scores}, topk_indices {topk_indices}")
                # print(f"rank {self.rank} load_balance_loss {load_balance_loss}")

            # 按目标 GPU 分组输入
            inputs_to_send = [[] for _ in range(self.world_size)]
            output_buffers = [[] for _ in range(self.world_size)]
            local_outputs = []

            batch_size = x.size(0)
            for i in range(batch_size):
                for j in range(self.top_k):
                    expert_idx = topk_indices[i, j].item()
                    target_rank = expert_idx // self.experts_per_gpu
                    score = topk_scores[i, j]
                    if target_rank == self.rank:
                        # 本地专家处理
                        local_expert_idx = expert_idx - self.start_expert_idx
                        local_output = self.local_experts[local_expert_idx](
                            x[i : i + 1]
                        )  # 处理单个样本
                        local_outputs.append((i, j, local_output.squeeze(0) * score))
                    else:
                        # 发送到其他 GPU
                        inputs_to_send[target_rank].append((i, j, x[i]))
                        # print(f"rank {self.rank} send to {target_rank} {i} {x[i]} {j}")

            # print(
            #    f"rank {self.rank} local_outputs {local_outputs} \ninputs_to_send {inputs_to_send}"
            # )

            # 同步接收输入
            recv_buffers = []
            for r in range(self.world_size):
                if r != self.rank and inputs_to_send[r]:
                    x_tensor = torch.stack([x for _, _, x in inputs_to_send[r]])
                    ctx_expert_idx = torch.tensor(
                        [[i, j] for i, j, _ in inputs_to_send[r]],
                        dtype=torch.long,
                        device=self.device,
                    )  # 对应的专家索引
                    send_tensor = torch.cat((ctx_expert_idx, x_tensor), dim=1)

                    num_inputs = torch.tensor(
                        [len(inputs_to_send[r])], dtype=torch.long, device=self.device
                    )
                    dist.broadcast(
                        num_inputs, src=self.rank
                    )  # 将对应的样本大小发给对应GPU的专家处理
                    dist.broadcast(send_tensor, src=self.rank)  # 将对应的样本发给对应GPU的专家处理
                    # print(f"rank {self.rank} send to {r} -> {send_tensor.shape}")
                if r != self.rank:
                    recv_num_inputs = torch.zeros(1, dtype=torch.long, device=self.device)
                    dist.broadcast(recv_num_inputs, src=r)
                    # print(f"rank {self.rank} recv num_inputs {recv_num_inputs} from {r}")
                    if recv_num_inputs.item() > 0:
                        recv_tensor = torch.zeros(
                            recv_num_inputs.item(), 2 + x.size(1), device=self.device
                        )
                        # 接收对应的样本,给对应GPU的专家处理
                        dist.broadcast(recv_tensor, src=r)
                        # print(f"rank {self.rank} recv recv_tensor {recv_tensor} <- from {r}")
                        recv_buffers.append((r, recv_tensor))

            # 处理接收到的样本
            for src_rank, recv_tensor in recv_buffers:
                for j in range(recv_tensor.size(0)):
                    ctx_expert_idx_i = int(recv_tensor[j : j + 1][0, 0].item())
                    ctx_expert_idx_j = int(recv_tensor[j : j + 1][0, 1].item())
                    expert_idx = topk_indices[ctx_expert_idx_i, ctx_expert_idx_j].item()
                    local_expert_idx = expert_idx - self.start_expert_idx
                    if local_expert_idx >= self.end_expert_idx:
                        print(
                            f"[WARNING] rank {self.rank} local_expert_idx {local_expert_idx} >= {self.end_expert_idx},don't belong to this rank"
                        )
                        continue
                    output = self.local_experts[local_expert_idx](
                        recv_tensor[j : j + 1][:, 2:]
                    )  # 处理单个样本 [1,input_size] -> [1,hidden_size]
                    output_buffers[src_rank].append(
                        (ctx_expert_idx_i, ctx_expert_idx_j, output.squeeze(0))
                    )

            # dist.barrier()

            # 发送输出回原 GPU
            for r in range(self.world_size):
                if r != self.rank and output_buffers[r]:
                    x_tensor = torch.stack([x for _, _, x in output_buffers[r]])
                    ctx_expert_idx = torch.tensor(
                        [[i, j] for i, j, _ in output_buffers[r]],
                        dtype=torch.long,
                        device=self.device,
                    )  # 对应的专家索引
                    send_tensor = torch.cat((ctx_expert_idx, x_tensor), dim=1)
                    dist.broadcast(send_tensor, src=self.rank)
                    # print(f"rank {self.rank} send to dst {r} {send_tensor.shape}")

            # 接收返回的输出
            final_topk_output = torch.zeros(
                batch_size, self.top_k, self.local_experts[0].fc.out_features, device=self.device
            )  # [batch_size, top_k, hidden_size]
            for r in range(self.world_size):
                if r != self.rank and inputs_to_send[r]:
                    num_outputs = len(inputs_to_send[r])
                    recv_tensor = torch.zeros(
                        num_outputs, 2 + final_topk_output.size(2), device=self.device
                    )
                    dist.broadcast(recv_tensor, src=r)
                    # print(
                    #    f"rank {self.rank} final_topk_output: {final_topk_output.shape} inputs_to_send: {len(inputs_to_send[r])}, recv_tensor: {recv_tensor.shape}"
                    # )
                    for j in range(recv_tensor.size(0)):
                        ctx_expert_idx_i = int(recv_tensor[j : j + 1][0, 0].item())
                        ctx_expert_idx_j = int(recv_tensor[j : j + 1][0, 1].item())
                        score = topk_scores[ctx_expert_idx_i, ctx_expert_idx_j]
                        out = recv_tensor[j : j + 1][:, 2:]
                        final_topk_output[ctx_expert_idx_i][ctx_expert_idx_j] += (
                            out.squeeze(0) * score
                        )  # Top-k 加权累加

            # 合并本地输出
            for i, j, out in local_outputs:
                final_topk_output[i][j] += out  # Top-k 加权累加

            # print(final_topk_output)
            # [batch_size, top_k, hidden_size] -> [batch_size, hidden_size]
            final_output = torch.mean(final_topk_output, dim=1)
            # final_output = torch.sum(final_topk_output, dim=1)
            # print(final_output)

            return final_output, load_balance_loss

    class All2AllMoE(MoE):
        def forward(self, x):
            batch_size = x.size(0)
            gate_scores = self.gating(x)  # [batch_size, num_experts]
            topk_scores, topk_indices = torch.topk(
                gate_scores, self.top_k, dim=-1
            )  # [batch_size, k]

            # 计算负载均衡损失
            load_balance_loss = self.compute_load_balance_loss(gate_scores, topk_indices)

            # 统计每个 GPU 需要处理的样本数
            counts_per_rank = torch.zeros(self.world_size, dtype=torch.long, device=self.device)
            for i in range(batch_size):
                for j in range(self.top_k):
                    expert_idx = topk_indices[i, j].item()
                    target_rank = expert_idx // self.experts_per_gpu
                    counts_per_rank[target_rank] += 1

            # 准备发送的输入张量和元数据
            send_tensors = [[] for _ in range(self.world_size)]
            send_indices = [[] for _ in range(self.world_size)]
            send_scores = [[] for _ in range(self.world_size)]
            for i in range(batch_size):
                for j in range(self.top_k):
                    expert_idx = topk_indices[i, j].item()
                    target_rank = expert_idx // self.experts_per_gpu
                    send_tensors[target_rank].append(x[i])
                    send_indices[target_rank].append(i)
                    send_scores[target_rank].append(topk_scores[i, j])

            # 将列表转换为张量列表，并填充到最大长度
            max_count = counts_per_rank.max().item()
            send_list = [
                torch.stack(send_tensors[r])
                if len(send_tensors[r]) > 0
                else torch.zeros(max_count, x.size(1), device=self.device)
                for r in range(self.world_size)
            ]
            recv_list = [
                torch.zeros(max_count, x.size(1), device=self.device)
                for _ in range(self.world_size)
            ]

            # 使用 all_to_all 分发输入
            dist.all_to_all(recv_list, send_list)

            # 处理接收到的输入
            local_outputs = torch.zeros(
                batch_size, self.local_experts[0].fc.out_features, device=self.device
            )
            for i in range(counts_per_rank[self.rank]):
                if i < recv_list[self.rank].size(0):
                    expert_idx = topk_indices[
                        send_indices[self.rank][i], i // counts_per_rank[self.rank] % self.top_k
                    ].item()
                    if self.start_expert_idx <= expert_idx < self.end_expert_idx:
                        local_expert_idx = expert_idx - self.start_expert_idx
                        output = self.local_experts[local_expert_idx](
                            recv_list[self.rank][i : i + 1]
                        )
                        orig_idx = send_indices[self.rank][i]
                        score = send_scores[self.rank][i]
                        local_outputs[orig_idx] += output.squeeze(0) * score

            # 使用 all_to_all 收集输出
            send_output_list = [
                torch.zeros(max_count, local_outputs.size(1), device=self.device)
                for _ in range(self.world_size)
            ]
            for i in range(counts_per_rank[self.rank]):
                if i < len(send_indices[self.rank]):
                    send_output_list[self.rank][i] = local_outputs[send_indices[self.rank][i]]

            recv_output_list = [
                torch.zeros(max_count, local_outputs.size(1), device=self.device)
                for _ in range(self.world_size)
            ]
            dist.all_to_all(recv_output_list, send_output_list)

            # 聚合最终输出
            final_output = torch.zeros(batch_size, local_outputs.size(1), device=self.device)
            for r in range(self.world_size):
                for i in range(counts_per_rank[r]):
                    if i < len(send_indices[r]):
                        orig_idx = send_indices[r][i]
                        score = send_scores[r][i]
                        final_output[orig_idx] += recv_output_list[r][i] * score
            return final_output, load_balance_loss

    # 专家并行训练函数
    def train_expert_parallel(
        rank,
        world_size,
        num_experts,
        input_size,
        hidden_size,
        data,
        labels,
        top_k,
        alpha=0.01,
        mode="brodcast",
    ):
        # 设置分布式环境
        setup_distributed(rank, world_size)

        # 参数设置
        device = torch.device(f"cuda:{rank}")

        # 将数据移动到设备
        data, labels = data.to(device), labels.to(device)

        # 初始化 MoE 模型
        if mode == "all2all":
            model = All2AllMoE(
                input_size, hidden_size, num_experts, rank, world_size, top_k, alpha
            ).to(device)
        else:
            model = MoE(input_size, hidden_size, num_experts, rank, world_size, top_k, alpha).to(
                device
            )

        # 定义损失函数和优化器
        criterion = nn.MSELoss()  # 假设回归任务
        # criterion = nn.CrossEntropyLoss()  # 分类任务
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 训练步骤
        model.train()
        for epoch in range(5):  # 模拟 5 个 epoch
            optimizer.zero_grad()

            # 前向传播
            output, load_balance_loss = model(data)
            task_loss = criterion(output, labels)

            # 总损失 = 任务损失 + 负载均衡损失
            loss = task_loss + load_balance_loss

            # 反向传播
            loss.backward()
            optimizer.step()

            # 打印损失
            dist.barrier()  # 同步所有进程
            if rank == 0:
                print(
                    f"Epoch {epoch+1}, Loss: {loss.item():.4f} Task Loss: {task_loss.item():.4f} Load Balance Loss: {load_balance_loss.item():.4f}"
                )

        cleanup()

    # 生成模拟数据
    batch_size = 4
    input_size = 2
    hidden_size = 6
    data = torch.randn(batch_size, input_size, requires_grad=True)
    labels = torch.randn(batch_size, hidden_size, requires_grad=True)  # 启用梯度

    # 启动进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=train_expert_parallel,
            args=(
                rank,
                world_size,
                num_experts,
                input_size,
                hidden_size,
                data,
                labels,
                top_k,
                alpha,
                mode,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
