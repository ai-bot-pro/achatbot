import os
import modal


app = modal.App("train-demo-ddp")

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
        print("需要至少 2 个 GPU 来演示分布式数据并行")
        return

    # 设置多进程启动方式
    torch.multiprocessing.set_start_method("spawn", force=True)

    print(f"Running Distributed Data Parallel with {world_size} GPUs")
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)


# modal run src/train/demo/ddp.py
@app.local_entrypoint()
def main():
    run.remote()


# 训练函数
def train_ddp(rank, world_size):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    import os

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
    hidden_size = 32
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

    # 初始化模型并包装为 DDP
    model = SimpleNN(input_size, hidden_size, output_size).to(device)
    # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    model = DDP(model, device_ids=[rank])

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
