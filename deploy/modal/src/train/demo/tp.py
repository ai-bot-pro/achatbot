# https://pytorch.org/tutorials/intermediate/TP_tutorial.html

import os
import modal


app = modal.App("train-demo-tp")

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
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    if world_size < 2:
        print("需要至少 2 个 GPU 来演示张量并行")
        return

    print(f"Running Tensor Parallelism with {world_size} GPUs")
    mp.spawn(train_tensor_parallel, args=(world_size,), nprocs=world_size, join=True)


# modal run src/train/demo/tp.py
@app.local_entrypoint()
def main():
    run.remote()


# 训练函数
def train_tensor_parallel(rank, world_size):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    import os

    # 初始化分布式环境
    def setup_distributed(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # 清理分布式环境
    def cleanup():
        dist.destroy_process_group()

    # 张量并行的线性层
    class TensorParallelLinear(nn.Module):
        def __init__(self, input_size, output_size, rank, world_size):
            super(TensorParallelLinear, self).__init__()
            self.rank = rank
            self.world_size = world_size

            # 将输出维度分割到多个设备上
            self.local_output_size = output_size // world_size
            assert output_size % world_size == 0, "Output size must be divisible by world_size"

            # 每个设备只持有部分权重
            self.weight = nn.Parameter(torch.randn(self.local_output_size, input_size))
            self.bias = nn.Parameter(torch.randn(self.local_output_size))

        def forward(self, x):
            # 本地计算：每个设备处理部分输出
            local_output = torch.matmul(x, self.weight.t()) + self.bias  # xw+b

            # 全局输出：通过 All-Gather 收集所有设备的输出
            global_output = [torch.zeros_like(local_output) for _ in range(self.world_size)]
            dist.all_gather(global_output, local_output)
            global_output = torch.cat(global_output, dim=-1)  # 拼接所有设备的输出

            return global_output

    # 张量并行的 MLP 层
    class TensorParallelMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, rank, world_size):
            super(TensorParallelMLP, self).__init__()
            self.rank = rank
            self.world_size = world_size

            # 第一层：按列分割隐藏维度
            assert hidden_size % world_size == 0, "hidden_size must be divisible by world_size"
            self.local_hidden_size = hidden_size // world_size
            self.fc1_weight = nn.Parameter(torch.randn(self.local_hidden_size, input_size))
            self.fc1_bias = nn.Parameter(torch.randn(self.local_hidden_size))
            self.relu = nn.ReLU()

            # 第二层：按行分割输出维度
            assert output_size % world_size == 0, "output_size must be divisible by world_size"
            self.local_output_size = output_size // world_size
            self.fc2_weight = nn.Parameter(torch.randn(self.local_output_size, hidden_size))
            self.fc2_bias = nn.Parameter(torch.randn(self.local_output_size))

        # @torch.no_grad()
        def forward(self, x):
            # 第一层计算：局部隐藏输出
            hidden = (
                torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
            )  # [batch_size, local_hidden_size]

            # All-Gather 收集所有设备的隐藏输出
            global_hidden = [torch.zeros_like(hidden) for _ in range(self.world_size)]
            dist.all_gather(global_hidden, hidden)
            global_hidden = torch.cat(global_hidden, dim=-1)  # [batch_size, hidden_size]
            global_hidden = self.relu(global_hidden)

            # 第二层计算：局部输出
            local_output = (
                torch.matmul(global_hidden, self.fc2_weight.t()) + self.fc2_bias
            )  # [batch_size, local_output_size]

            # All-Gather 收集所有设备的输出
            global_output = [torch.zeros_like(local_output) for _ in range(self.world_size)]
            dist.all_gather(global_output, local_output)
            global_output = torch.cat(global_output, dim=-1)  # [batch_size, output_size]

            return global_output

    # 设置分布式环境
    print(f"setup_distributed rank:{rank} world_size:{world_size}")
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank} using device: {device}")

    # 参数设置
    input_size = 16
    hidden_size = 32  # 必须能被 world_size 整除 # 线性层没有隐藏层
    output_size = 8  # 必须能被 world_size 整除
    batch_size = 4

    # 初始化模型
    model = TensorParallelLinear(input_size, output_size, rank, world_size).to(device)
    # model = TensorParallelMLP(input_size, hidden_size, output_size, rank, world_size).to(device)

    # 生成模拟输入数据
    x = torch.randn(batch_size, input_size, requires_grad=True).to(device)
    target = torch.randn(batch_size, output_size, requires_grad=True).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 回归任务
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练步骤
    model.train()
    optimizer.zero_grad()

    # 前向传播
    output = model(x)
    loss = criterion(output, target)

    # 反向传播
    loss.backward()
    optimizer.step()

    # 同步并打印损失
    dist.barrier()
    # if rank == 0:
    print(f"rank {rank}, Loss: {loss.item():.4f}")

    cleanup()
