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
def run(model_name="linear"):
    import torch
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    if world_size < 2:
        print("需要至少 2 个 GPU 来演示张量并行")
        return

    train_pipeline_parallel(world_size)


# modal run src/train/demo/tp.py
@app.local_entrypoint()
def main(model_name="linear"):
    run.remote(model_name)


# 训练函数
def train_pipeline_parallel(world_size):
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from torch.multiprocessing import Process

    # 初始化分布式环境
    def setup_distributed(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # 清理分布式环境
    def cleanup():
        dist.destroy_process_group()

    # 定义模型的阶段
    class Stage1(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Stage1, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()

        def forward(self, x):  # [batch_size, input_size]
            return self.relu(self.fc1(x))  # [batch_size, hidden_size]

    class Stage2(nn.Module):
        def __init__(self, hidden_size, output_size):
            super(Stage2, self).__init__()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):  # [batch_size, hidden_size]
            return self.fc2(x)  # [batch_size, output_size]

    # 流水线训练函数
    def train_pipeline(rank, world_size, data, labels):
        # 设置分布式环境
        setup_distributed(rank, world_size)

        # 参数设置
        input_size = 16
        hidden_size = 32
        output_size = 2
        device = torch.device(f"cuda:{rank}")

        # 将数据和标签移动到对应设备
        data, labels = data.to(device), labels.to(device)

        # 根据 rank 分配阶段
        if rank == 0:
            model = Stage1(input_size, hidden_size).to(device)
        elif rank == 1:
            model = Stage2(hidden_size, output_size).to(device)

        # 定义优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 训练步骤
        num_microbatches = 4  # 微批次数量
        microbatch_size = data.size(0) // num_microbatches

        for epoch in range(5):  # 模拟 5 个 epoch
            optimizer.zero_grad()
            microbatch_size = data.size(0) // num_microbatches

            # 前向传播：微批次处理
            intermediates = []
            outputs = []
            for i in range(num_microbatches):
                start = i * microbatch_size
                end = start + microbatch_size
                micro_data = data[start:end]
                micro_labels = labels[start:end]

                if rank == 0:
                    # 第一阶段：计算并发送中间结果
                    intermediate = model(micro_data)
                    intermediate = intermediate.detach().requires_grad_(True)  # 保留梯度
                    dist.send(tensor=intermediate, dst=1)
                    intermediates.append(intermediate)
                elif rank == 1:
                    # 第二阶段：接收中间结果并计算输出
                    intermediate = torch.zeros(
                        microbatch_size, hidden_size, device=device, requires_grad=True
                    )  # 注意这里需要申请要求保留梯度, 因为需要反向传播计算梯度
                    dist.recv(tensor=intermediate, src=0)
                    output = model(intermediate)  # 确保 intermediate 参与计算图
                    outputs.append((output, micro_labels, intermediate))

            # 反向传播：计算损失并传递梯度
            # 最后一步用来计算loss, 反向传播后，然后把计算的梯度发送给上一阶段来反向传播计算梯度，依次类推
            if rank == 1:
                loss_fn = nn.CrossEntropyLoss(reduction="sum")  # 使用 sum 以累积微批次损失
                total_loss = 0
                for output, micro_labels, intermediate in outputs:
                    loss = loss_fn(output, micro_labels)
                    total_loss += loss
                    loss.backward()  # 为每个微批次单独反向传播，计算梯度,确保 intermediate.grad 生成

                    # 发送中间梯度给 Stage 1
                    assert intermediate.grad is not None, "intermediate.grad is None in Stage 2"
                    dist.send(tensor=intermediate.grad, dst=0)

                total_loss /= num_microbatches  # 平均损失
                print(f"Epoch {epoch+1} loss: {total_loss}")
            elif rank == 0:
                for intermediate in intermediates:
                    grad = torch.zeros_like(intermediate).to(device)
                    dist.recv(tensor=grad, src=1)
                    intermediate.backward(grad)  # 应用梯度

            optimizer.step()

            # 同步并打印损失
            dist.barrier()
            if rank == 0:
                print(f"Epoch {epoch+1} finished on Rank {rank}")
        cleanup()

    # 生成模拟数据
    batch_size = 16
    input_size = 16
    output_size = 2
    data = torch.randn(batch_size, input_size, requires_grad=True)
    labels = torch.randint(0, output_size, (batch_size,))

    print(f"Running Pipeline Parallelism with {world_size} GPUs")

    # 启动进程
    processes = []
    for rank in range(world_size):
        p = Process(target=train_pipeline, args=(rank, world_size, data, labels))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
