import os
import modal


app = modal.App("train-demo-pp")

demo_image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")

MOCK_EPOCH = 5  # 模拟 5 个 epoch


@app.function(
    image=demo_image,
    gpu=os.getenv("IMAGE_GPU", "T4:2"),
    retries=0,
    # how long should we stay up with no requests?
    scaledown_window=15 * 60,
)
def run(mode):
    import torch
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    if world_size < 2:
        print("需要至少 2 个 GPU 来演示张量并行")
        return

    if mode == "async":
        train_async_pipeline_parallel(world_size)
    else:
        train_pipeline_parallel(world_size)


"""
modal run src/train/demo/pp.py
modal run src/train/demo/pp.py --mode async
"""


@app.local_entrypoint()
def main(mode=""):
    run.remote(mode)


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

        for epoch in range(MOCK_EPOCH):
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
            # 最后一步用来计算loss, 反向传播后，把计算的梯度发送给上一阶段来反向传播计算梯度，依次类推
            if rank == 1:
                loss_fn = nn.CrossEntropyLoss(reduction="sum")  # 使用 sum 以累积微批次损失
                total_loss = 0
                for output, micro_labels, intermediate in outputs:
                    # print(f"output: {output} {output.shape}, {micro_labels} {micro_labels.shape}")
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
    output_size = 2  # 二分类
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


def train_async_pipeline_parallel(world_size):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.multiprocessing import Process, Queue
    import os
    import queue
    from datetime import timedelta

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

        def forward(self, x):
            return self.relu(self.fc1(x))

    class Stage2(nn.Module):
        def __init__(self, hidden_size, output_size):
            super(Stage2, self).__init__()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            return self.fc2(x)

    # 异步流水线训练函数
    def train_async_pipeline(
        rank, input_size, world_size, num_microbatches, microbatch_size, data_queue, result_queue
    ):
        # 设置分布式环境
        setup_distributed(rank, world_size)

        # 参数设置
        hidden_size = 32
        output_size = 2  # 二分类
        device = torch.device(f"cuda:{rank}")

        # 根据 rank 分配阶段
        if rank == 0:
            model = Stage1(input_size, hidden_size).to(device)
        elif rank == 1:
            model = Stage2(hidden_size, output_size).to(device)

        # 定义优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 异步流水线处理
        model.train()
        for epoch in range(MOCK_EPOCH):
            optimizer.zero_grad()

            if rank == 0:
                intermediates = []
                # Stage 1: 处理输入并发送中间结果
                for i in range(num_microbatches):
                    try:
                        micro_data = data_queue.get(timeout=1)  # 从队列获取微批次数据
                        intermediate = model(micro_data.to(device))  # 前向计算
                        # print(f"isend intermediate: {intermediate} {intermediate.shape}")
                        dist.isend(tensor=intermediate, dst=1)  # 异步发送
                        intermediate = intermediate.detach().requires_grad_(
                            True
                        )  # 断开计算图，但保留梯度
                        intermediates.append(intermediate)  # 保存中间结果用于梯度传递
                    except queue.Empty:
                        print("no more data, break")
                        break

                # 接收并应用梯度
                for i in range(num_microbatches):
                    if i >= len(intermediates):
                        break
                    intermediate = intermediates[i]
                    grad = torch.zeros_like(intermediate).to(device)
                    dist.irecv(tensor=grad, src=1).wait()  # 异步接收梯度
                    # print(f"irecv intermediate.grad: {grad} {grad.shape}")
                    intermediate.backward(grad)

            elif rank == 1:
                total_loss = 0
                # Stage 2: 接收中间结果并计算输出
                for i in range(num_microbatches):
                    intermediate = torch.zeros(
                        microbatch_size, hidden_size, requires_grad=True, device=device
                    )
                    try:
                        # is_ok = dist.irecv(tensor=intermediate, src=0).wait()
                        is_ok = dist.irecv(tensor=intermediate, src=0).wait(
                            timeout=timedelta(seconds=3)
                        )
                    # except RuntimeError as e:
                    except RuntimeError:
                        # print(f"Error during irecv.wait: {e}")
                        is_ok = False

                    if not is_ok:
                        print("wait timeout, no data, next")
                        break
                    # print(f"irecv intermediate: {intermediate} {intermediate.shape}")
                    output = model(intermediate)

                    # 计算损失（假设标签为随机生成）
                    loss_fn = nn.CrossEntropyLoss()
                    micro_labels = torch.randint(0, output_size, (microbatch_size,)).to(device)
                    loss = loss_fn(output, micro_labels)
                    total_loss += loss.item()
                    loss.backward()

                    # 异步发送梯度回 Stage 1
                    assert intermediate.grad is not None, "intermediate.grad is None in Stage 2"
                    # print(f"isend intermediate.grad: {intermediate.grad} {intermediate.grad.shape}")
                    dist.isend(tensor=intermediate.grad, dst=0)
                total_loss = total_loss / num_microbatches
                print(f"Epoch {epoch+1} loss: {total_loss}\n")

            optimizer.step()
            torch.cuda.synchronize()  # 确保所有操作完成
            if rank == 0:
                print(f"Epoch {epoch+1} finished on Rank {rank}\n")

        cleanup()

    # 生成模拟数据
    batch_size = 4  # 微批次总和
    input_size = 16
    num_microbatches = 4  # 每个微批次大小
    data = torch.randn(batch_size, input_size)

    # 创建数据队列和结果队列
    data_queue = Queue()
    result_queue = Queue()

    # 将数据分片放入队列
    microbatch_size = batch_size // num_microbatches  # 假设 4 个微批次
    for _ in range(MOCK_EPOCH):
        for i in range(0, batch_size, microbatch_size):
            micro_data = data[i : i + microbatch_size]
            data_queue.put(micro_data)

    print(f"Running Asynchronous Pipeline Parallelism with {world_size} GPUs")

    # 启动进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=train_async_pipeline,
            args=(
                rank,
                input_size,
                world_size,
                num_microbatches,
                microbatch_size,
                data_queue,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
