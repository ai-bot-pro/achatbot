import os
import modal


app = modal.App("train-demo-sp")

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

    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    if world_size < 2:
        print("需要至少 2 个 GPU 来演示分布式数据并行")
        return

    print(f"Running Sequence Parallelism on {world_size} GPUs")
    train_sp(world_size)


# modal run src/train/demo/sp.py
@app.local_entrypoint()
def main():
    run.remote()


# 训练函数
def train_sp(world_size):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.multiprocessing import Process
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

    # 定义简单的 Transformer 层（支持序列并行）
    class SequenceParallelTransformer(nn.Module):
        def __init__(self, embed_dim, num_heads, rank, world_size, dropout=0.01):
            super(SequenceParallelTransformer, self).__init__()
            self.rank = rank
            self.world_size = world_size
            self.embed_dim = embed_dim

            # 自注意力层
            self.attent_norm = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads)
            self.norm = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim)
            )
            self.norm_ffn = nn.LayerNorm(embed_dim)
            self.attent_dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x 的形状: [local_seq_len, batch_size, embed_dim]
            local_seq_len, batch_size, _ = x.size()
            x = self.attent_norm(x)

            # 在序列并行中，每个 GPU 只处理部分序列
            # 使用 all_gather 收集完整的序列用于注意力计算
            full_seq = [
                torch.zeros(local_seq_len, batch_size, self.embed_dim, device=x.device)
                for _ in range(self.world_size)
            ]
            dist.all_gather(full_seq, x)
            full_seq = torch.cat(full_seq, dim=0)  # [seq_len, batch_size, embed_dim]

            # 自注意力计算（需要完整序列）
            attn_output, _ = self.attn(full_seq, full_seq, full_seq)
            attn_output = attn_output[
                self.rank * local_seq_len : (self.rank + 1) * local_seq_len
            ]  # 取回本地部分
            self.attent_dropout(attn_output)

            # 残差连接和归一化
            x = self.norm(x + attn_output)

            # 前馈网络（局部计算，无需通信）
            ffn_output = self.ffn(x)
            output = self.norm_ffn(x + ffn_output)

            return output

    def train_sequence_parallel(rank, world_size, data, labels):
        setup_distributed(rank, world_size)

        # 参数设置
        num_heads = 8
        seq_len, batch_size, embed_dim = data.size()
        device = torch.device(f"cuda:{rank}")

        # 将数据移动到设备并分割序列
        local_seq_len = seq_len // world_size
        data = data.to(device)
        labels = labels.to(device)
        local_data = data[rank * local_seq_len : (rank + 1) * local_seq_len]  # 分割序列
        local_labels = labels[rank * local_seq_len : (rank + 1) * local_seq_len]

        # 初始化模型
        model = SequenceParallelTransformer(embed_dim, num_heads, rank, world_size).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 训练步骤
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()

            # 前向传播
            output = model(local_data)
            loss = criterion(output, local_labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 同步并打印损失
            dist.barrier()
            if rank == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        cleanup()

    # 生成模拟数据
    batch_size = 4
    seq_len = 20  # 总序列长度，必须能被 world_size 整除
    embed_dim = 64
    data = torch.randn(seq_len, batch_size, embed_dim)
    labels = torch.randn(seq_len, batch_size, embed_dim)  # 假设回归任务

    processes = []
    for rank in range(world_size):
        p = Process(target=train_sequence_parallel, args=(rank, world_size, data, labels))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
