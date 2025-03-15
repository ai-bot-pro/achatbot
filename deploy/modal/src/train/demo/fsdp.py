import os
import modal


app = modal.App("train-demo-fsdp")

demo_image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


@app.function(
    image=demo_image,
    gpu=os.getenv("IMAGE_GPU", "T4:2"),
    retries=0,
    # how long should we stay up with no requests?
    scaledown_window=15 * 60,
)
def run():
    import torch
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    if world_size < 2:
        print("需要至少 2 个 GPU 来演示全分片数据并行")
        return

    print(f"Running Fully Sharded Data Parallel with {world_size} GPUs")
    mp.spawn(train_fsdp, args=(world_size,), nprocs=world_size, join=True)


# modal run src/train/demo/fsdp.py
@app.local_entrypoint()
def main():
    run.remote()


# 训练函数
def train_fsdp(rank, world_size):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    import os
    import functools

    # 初始化分布式环境
    def setup_distributed(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        # https://pytorch.org/docs/stable/distributed.html gpu use nccl
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # 清理分布式环境
    def cleanup():
        dist.destroy_process_group()

    # 定义一个简单的神经网络
    class SimpleNN(nn.Module):
        """MLP"""

        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # 生成模拟数据
    def generate_dummy_data(num_samples, input_size):
        X = torch.randn(num_samples, input_size)
        y = torch.randint(0, 2, (num_samples,))
        return X, y

    # 设置分布式环境
    print(f"setup_distributed rank:{rank} world_size:{world_size}")
    setup_distributed(rank, world_size)

    # 参数设置
    input_size = 16
    hidden_size = 1024  # 增加隐藏层大小以展示分片效果
    output_size = 2
    batch_size = 8
    num_samples = 64
    num_epochs = 5

    # 设备设置
    device = torch.device(f"cuda:{rank}")

    # 生成数据
    X, y = generate_dummy_data(num_samples, input_size)
    dataset = TensorDataset(X, y)

    # 使用 DistributedSampler 分配数据
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 初始化模型
    model = SimpleNN(input_size, hidden_size, output_size).to(device)

    # 定义自动包装策略（基于参数大小）
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        module=model,
        recurse=True,
        nonwrapped_numel=0,
        min_num_params=int(1e3),
    )
    # 使用 FSDP 包装模型
    # https://pytorch.org/docs/stable/fsdp.html
    """
    - 准备工作: 定义了一个基于参数数量的自动包装策略，这个策略规定了哪些模型模块需要被 FSDP 包装。
    - FSDP 应用: 使用这个策略，利用 FSDP 类对 SimpleNN 模型进行包装，并且选择了 FULL_SHARD 策略来尽可能减少显存占用。
    - 参数的意义: min_num_params=int(1e3)决定了模型需要多大，才会被单独包装。hidden_size = 1024，SimpleNN包括两层线性层，fc1 和 fc2，fc1 参数量 16*1024=16384 > 1000，会被单独包装，fc2参数量 1024*2=2048 > 1000，也会被单独包装。其他没有被单独包装的参数，会被统一管理。
    """
    model = FSDP(
        model,
        device_id=rank,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,  # ZeRO-3
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据分配不同
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        # 同步所有进程并打印损失
        dist.barrier()
        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    cleanup()
