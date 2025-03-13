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
    scaledown_window=15 * 60,
)
def run(model_name="linear"):
    import torch
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    if world_size < 2:
        print("需要至少 2 个 GPU 来演示张量并行")
        return

    print(f"Running model {model_name} Tensor Parallelism with {world_size} GPUs")
    torch.multiprocessing.set_start_method("spawn", force=True)
    mp.spawn(
        train_tensor_parallel,
        args=(
            world_size,
            model_name,
        ),
        nprocs=world_size,
        join=True,
    )


"""
# linear
modal run src/train/demo/tp.py
# mlp use all_reduce
modal run src/train/demo/tp.py --model-name mlp
# mlp use all_gather
modal run src/train/demo/tp.py --model-name mlp_all_gather
# mha
modal run src/train/demo/tp.py --model-name mha
# embedding
modal run src/train/demo/tp.py --model-name embedding
# mha_mlp
modal run src/train/demo/tp.py --model-name mha_mlp
"""


@app.local_entrypoint()
def main(model_name="linear"):
    run.remote(model_name)


# 训练函数
def train_tensor_parallel(rank, world_size, model_name="linear"):
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
        def __init__(self, seq_len, output_size, rank, world_size):
            super(TensorParallelLinear, self).__init__()
            self.rank = rank
            self.world_size = world_size

            # 将输出维度分割到多个设备上
            self.local_output_size = output_size // world_size
            assert output_size % world_size == 0, "Output size must be divisible by world_size"

            # 每个设备只持有部分权重
            self.weight = nn.Parameter(torch.randn(self.local_output_size, seq_len))
            self.bias = nn.Parameter(torch.randn(self.local_output_size))

        def forward(self, x):
            # 本地计算：每个设备处理部分输出
            local_output = torch.matmul(x, self.weight.t()) + self.bias  # XxW^T+B

            # 全局输出：通过 All-Gather 收集所有设备的输出
            global_output = [torch.zeros_like(local_output) for _ in range(self.world_size)]
            dist.all_gather(global_output, local_output)
            global_output = torch.cat(global_output, dim=-1)  # 拼接所有设备的输出

            return global_output

    # 张量并行的 MLP 层(使用 all_gather/all_reduce)
    class TensorParallelMLP(nn.Module):
        def __init__(
            self,
            seq_len,
            embed_dim,
            output_size,
            rank,
            world_size,
            communication="all_reduce",
        ):
            super(TensorParallelMLP, self).__init__()
            self.rank = rank
            self.world_size = world_size
            self.communication = communication

            if communication == "all_reduce":
                # 第一层：按列分割隐藏维度
                assert embed_dim % world_size == 0, "embed_dim must be divisible by world_size"
                self.local_embed_dim = embed_dim // world_size
                self.fc1_weight = nn.Parameter(torch.randn(seq_len, self.local_embed_dim))
                self.fc1_bias = nn.Parameter(torch.randn(self.local_embed_dim))
                self.relu = nn.ReLU()

                # 第二层：按行分割隐藏维度
                self.fc2_weight = nn.Parameter(torch.randn(self.local_embed_dim, output_size))
                self.fc2_bias = nn.Parameter(torch.randn(output_size))
            else:  # default all_gather
                # 第一层：按列分割隐藏维度
                assert embed_dim % world_size == 0, "embed_dim must be divisible by world_size"
                self.local_embed_dim = embed_dim // world_size
                self.fc1_weight = nn.Parameter(torch.randn(seq_len, self.local_embed_dim))
                self.fc1_bias = nn.Parameter(torch.randn(self.local_embed_dim))
                self.relu = nn.ReLU()

                # 第二层：按列分割输出维度
                assert output_size % world_size == 0, "output_size must be divisible by world_size"
                self.local_output_size = output_size // world_size
                self.fc2_weight = nn.Parameter(torch.randn(embed_dim, self.local_output_size))
                self.fc2_bias = nn.Parameter(torch.randn(self.local_output_size))

        def forward(self, x):  # x: [batch_size, seq_len]
            if self.communication == "all_reduce":
                return self.all_reduce_forward(x)
            return self.all_gather_forward(x)

        def all_reduce_forward(self, x):
            """
            1次 all_reduce
            减少通信量的场景，尤其在输入维度较大时
            """
            # 第一层计算：局部隐藏输出
            local_hidden = (  # [batch_size, seq_len] x [seq_len, local_embed_dim] + [local_embed_dim]
                torch.matmul(x, self.fc1_weight) + self.fc1_bias
            )  # [batch_size, local_embed_dim]

            # 第二层计算：局部输出
            output = (  # [batch_size, local_embed_dim] x [local_embed_dim, output_size] + [output_size]
                torch.matmul(local_hidden, self.fc2_weight) + self.fc2_bias
            )  # [batch_size, output_size]

            # All-Reduce 汇总所有设备的输出广播
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

            return output

        def all_gather_forward(self, x):
            """2次 all_gather"""
            # 第一层计算：局部隐藏输出
            local_hidden = (  # [batch_size, seq_len] x [seq_len, local_embed_dim] + [local_embed_dim]
                torch.matmul(x, self.fc1_weight) + self.fc1_bias
            )  # [batch_size, local_embed_dim]

            # All-Gather 收集所有设备的隐藏输出
            global_hidden = [torch.zeros_like(local_hidden) for _ in range(self.world_size)]
            dist.all_gather(global_hidden, local_hidden)
            global_hidden = torch.cat(global_hidden, dim=-1)  # [batch_size, embed_dim]
            global_hidden = self.relu(global_hidden)

            # 第二层计算：局部输出
            local_output = (  # [batch_size, embed_dim] x [embed_dim, local_output_size] + [local_output_size]
                torch.matmul(global_hidden, self.fc2_weight) + self.fc2_bias
            )  # [batch_size, local_output_size]

            # All-Gather 收集所有设备的输出
            global_output = [torch.zeros_like(local_output) for _ in range(self.world_size)]
            dist.all_gather(global_output, local_output)
            global_output = torch.cat(global_output, dim=-1)  # [batch_size, output_size]

            return global_output

    # 张量并行的多头注意力机制
    class TensorParallelMultiheadAttention(nn.Module):
        def __init__(self, embed_dim, num_heads, rank, world_size):
            super(TensorParallelMultiheadAttention, self).__init__()
            self.rank = rank
            self.world_size = world_size
            self.num_heads = num_heads
            self.embed_dim = embed_dim

            # 每个设备分担部分头
            assert num_heads % world_size == 0, "num_heads must be divisible by world_size"
            self.local_heads = num_heads // world_size
            self.head_dim = embed_dim // num_heads

            # 分片 Q, K, V 的线性层
            self.qkv_dim = self.local_heads * self.head_dim  # 每个设备处理的维度
            self.q_proj = nn.Linear(embed_dim, self.qkv_dim)
            self.k_proj = nn.Linear(embed_dim, self.qkv_dim)
            self.v_proj = nn.Linear(embed_dim, self.qkv_dim)
            self.out_proj = nn.Linear(self.qkv_dim * world_size, embed_dim)

        def forward(self, x):
            batch_size, seq_len, _ = x.size()

            # 计算 Q, K, V
            q = self.q_proj(x)  # [batch_size, seq_len, local_heads * head_dim]
            k = self.k_proj(x)
            v = self.v_proj(x)

            # 重塑为多头格式
            q = q.view(batch_size, seq_len, self.local_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.local_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.local_heads, self.head_dim).transpose(1, 2)

            # 注意力计算
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, v)

            # 重塑回原始维度
            context = (
                context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.qkv_dim)
            )  # [batch_size, seq_len, qkv_dim]

            # All-Gather 收集所有设备的输出
            global_context = [torch.zeros_like(context) for _ in range(self.world_size)]
            dist.all_gather(global_context, context)  # [rank0_context, rank1_context, ...]
            global_context = torch.cat(
                global_context, dim=-1
            )  # 拼接所有头的输出 [batch_size, seq_len, qkv_dim*world_size]

            # 输出投影
            output = self.out_proj(
                global_context
            )  # [batch_size, seq_len, qkv_dim*world_size] -> [batch_size, seq_len, embed_dim]

            return output

    # Transformer 层
    class TPMHATransformerLayer(nn.Module):
        def __init__(self, embed_dim, num_heads, rank, world_size):
            super(TPMHATransformerLayer, self).__init__()
            self.attn = TensorParallelMultiheadAttention(embed_dim, num_heads, rank, world_size)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )
            self.att_norm = nn.LayerNorm(embed_dim)
            self.ffn_norm = nn.LayerNorm(embed_dim)

        def forward(self, x):
            x = x + self.attn(self.att_norm(x))
            x = x + self.ffn(self.ffn_norm(x))
            return x

    # 张量并行的 Embedding 层
    class TensorParallelEmbedding(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, rank, world_size):
            super(TensorParallelEmbedding, self).__init__()
            self.rank = rank
            self.world_size = world_size
            self.embedding_dim = embedding_dim

            # 按词汇表维度分割
            assert (
                num_embeddings % world_size == 0
            ), "num_embeddings must be divisible by world_size"
            self.local_num_embeddings = num_embeddings // world_size

            # 每个设备持有部分词汇的嵌入
            self.embedding = nn.Embedding(self.local_num_embeddings, embedding_dim)

            # 计算本地词汇的起始和结束索引
            self.start_idx = rank * self.local_num_embeddings
            self.end_idx = (rank + 1) * self.local_num_embeddings

        def forward(self, input_ids):
            # 将输入 ID 映射到本地范围
            batch_size, seq_len = input_ids.size()
            mask = (input_ids >= self.start_idx) & (input_ids < self.end_idx)
            local_input_ids = input_ids.clone()
            local_input_ids[~mask] = 0  # 非本地 ID 设为 0（假设 0 在本地范围）
            local_input_ids[mask] -= self.start_idx  # 调整到本地索引

            # 本地嵌入计算
            local_output = self.embedding(local_input_ids)  # [batch_size, seq_len, embedding_dim]

            # All-Gather 收集所有设备的输出
            global_output = [torch.zeros_like(local_output) for _ in range(self.world_size)]
            dist.all_gather(global_output, local_output)

            # 根据输入 ID 选择正确的输出
            output = torch.zeros_like(local_output)
            for i in range(self.world_size):
                start = i * self.local_num_embeddings
                end = (i + 1) * self.local_num_embeddings
                mask = (input_ids >= start) & (input_ids < end)
                output[mask] = global_output[i][mask]

            return output

    # 设置分布式环境
    print(f"setup_distributed rank:{rank} world_size:{world_size}")
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank} using device: {device}")

    batch_size = 4
    seq_len = 10
    # 初始化模型
    if model_name == "mlp":
        # 参数设置
        embed_dim = 64  # 必须能被 world_size 整除
        output_size = 8  # 必须能被 world_size 整除
        model = TensorParallelMLP(seq_len, embed_dim, output_size, rank, world_size).to(device)
        # 生成模拟输入数据
        x = torch.randn(batch_size, seq_len, requires_grad=True).to(device)
        # target = torch.randn(batch_size, output_size, requires_grad=True).to(device)
    elif model_name == "mlp_all_gather":
        # 参数设置
        embed_dim = 64  # 必须能被 world_size 整除
        output_size = 8  # 必须能被 world_size 整除
        model = TensorParallelMLP(
            seq_len, embed_dim, output_size, rank, world_size, communication="all_gather"
        ).to(device)
        # 生成模拟输入数据
        x = torch.randn(batch_size, seq_len, requires_grad=True).to(device)
        # target = torch.randn(batch_size, output_size, requires_grad=True).to(device)
    elif model_name == "mha":
        # 参数设置
        embed_dim = 64  # embed_dim
        num_heads = 8  # 必须能被 world_size 整除
        model = TensorParallelMultiheadAttention(embed_dim, num_heads, rank, world_size).to(device)
        # 生成模拟输入数据
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True).to(device)
    elif model_name == "embedding":
        # 参数设置
        embed_dim = 64
        num_embeddings = 26  # 词汇表大小，必须能被 world_size 整除, 字母表大小
        model = TensorParallelEmbedding(num_embeddings, embed_dim, rank, world_size).to(device)
        # 生成模拟输入数据
        x = torch.randint(0, num_embeddings, (batch_size, seq_len)).to(device)
    elif model_name == "mha_mlp":  # no embedding + tp mha + mlp no tp
        # 参数设置
        embed_dim = 64  # embed_dim
        num_heads = 8  # 必须能被 world_size 整除
        model = TPMHATransformerLayer(embed_dim, num_heads, rank, world_size).to(device)
        # 生成模拟输入数据
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True).to(device)
    else:  # 默认线性层，线性层没有隐藏层
        # 参数设置
        output_size = 8  # 必须能被 world_size 整除
        model = TensorParallelLinear(seq_len, output_size, rank, world_size).to(device)
        # 生成模拟输入数据
        x = torch.randn(batch_size, seq_len, requires_grad=True).to(device)
        # target = torch.randn(batch_size, output_size, requires_grad=True).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 回归任务
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练步骤
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()

        # 前向传播
        output = model(x)
        target = torch.randn_like(output)  # 模拟目标
        target.requires_grad = True
        loss = criterion(output, target.to(device))

        # 反向传播
        loss.backward()
        optimizer.step()

        # 同步并打印损失
        dist.barrier()
        # print(f"rank {rank}, Loss: {loss.item():.4f}")
        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    cleanup()
