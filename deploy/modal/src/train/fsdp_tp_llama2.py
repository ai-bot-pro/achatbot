# https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py


import os
import modal


app = modal.App("train-fsdp-tp-llama2")

demo_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands("git clone https://github.com/pytorch/examples.git /pytorch_example")
    .pip_install("torch")
)


"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

 We use a simple diagram to illustrate below:

======================================================================
------------       ------------       ------------       ------------
| Host 1   |       | Host 2   |       |          |       | Host N   |
| 8 GPUs   |       | 8 GPUs   |       |          |       | 8 GPUs   |
|          |       |          |       |    ...   |       |          |
| (TP)     |       | (TP)     |       |          |       | (TP)     |
|[0,1,..,7]|       |[8,9..,15]|       |          |       |[8N-8,8N-7|
|          |       |          |       |          |       | .., 8N-1]|
|          |       |          |       |          |       |          |
------------       ------------       ------------       ------------
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
======================================================================

More details can be seen in the PyTorch tutorials:
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
"""

"""
modal run src/train/fsdp_tp_llama2.py 
"""


@app.function(
    image=demo_image,
    gpu=os.getenv("IMAGE_GPU", "T4:4"),
    retries=0,
    # how long should we stay up with no requests?
    scaledown_window=15 * 60,
)
def run():
    import torch
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    if world_size < 4:
        print("需要至少 4 个 GPU 来演示分布式数据并行")
        return

    # 设置多进程启动方式
    torch.multiprocessing.set_start_method("spawn", force=True)

    print(f"Running FSDP + TP training llama2 with {world_size} GPUs")
    mp.spawn(train_fsdp_tp_llama2, args=(world_size,), nprocs=world_size, join=True)


def train_fsdp_tp_llama2(rank, world_size, tp_size=2, num_samples=64):
    import sys
    import os

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed._tensor import Shard, Replicate
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
        PrepareModuleInput,
        SequenceParallel,
    )
    import torch.nn.functional as F

    sys.path.insert(1, "/pytorch_example/distributed/tensor_parallelism")
    from log_utils import rank_log, get_logger
    from llama2_model import Transformer, ModelArgs

    # 初始化分布式环境
    def setup_distributed(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # 清理分布式环境
    def cleanup():
        dist.destroy_process_group()

    setup_distributed(rank, world_size)
    logger = get_logger()

    print(f"Starting PyTorch 2D (FSDP + TP) example on rank {rank}")
    assert (
        world_size % tp_size == 0
    ), f"World size {world_size} needs to be divisible by TP size {tp_size}"

    # create a sharding plan based on the given world_size.
    dp_size = world_size // tp_size

    # Create a device mesh with 2 dimensions.
    # First dim is the data parallel dimension
    # Second dim is the tensor parallel dimension.
    device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

    rank_log(rank, logger, f"Device Mesh 2D DP{dp_size}TP{tp_size} created: {device_mesh=}")
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    # For TP, input needs to be same across all TP ranks.
    # while for SP, input can be different across all ranks.
    # We will use dprank for setting the random seed
    # to mimic the behavior of the dataloader.
    dprank = dp_mesh.get_local_rank()

    # like: https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json but small params to train
    # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
    simple_llama2_config = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)

    model = Transformer.from_model_args(simple_llama2_config).to("cuda")
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    rank_log(rank, logger, f"{model_million_params} M parameters")

    # init model weights
    model.init_weights()

    # parallelize the first embedding and the last linear out projection
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
        },
    )

    for layer_id, transformer_block in enumerate(model.layers):
        layer_tp_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

        # Custom parallelization plan for the model
        parallelize_module(
            module=transformer_block, device_mesh=tp_mesh, parallelize_plan=layer_tp_plan
        )

    # Init FSDP using the dp device mesh
    sharded_model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)

    rank_log(rank, logger, f"Model after parallelization {sharded_model=}\n")

    # Create an optimizer for the parallelized and sharded model.
    lr = 3e-3
    rank_log(rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
    optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=True)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop:
    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    rank_log(rank, logger, "\nStarting 2D training...")
    num_iterations = 10

    batch_size = 8
    seq_len = 1024
    for i in range(num_iterations):
        # seeding with dprank to ensure identical inputs for TP groups
        torch.manual_seed(i + dprank)
        inp = torch.randint(
            simple_llama2_config.vocab_size, (batch_size, seq_len), device="cuda"
        )  # token ids range [0,vocab_size) shape [B,S]

        output = sharded_model(inp)  # [B,S,vocab_size]
        logits = output.view(-1, output.size(-1))  # [BxS,vocab_size]
        target = torch.randn_like(logits)  # just test
        # loss = F.cross_entropy(logits, target, ignore_index=-1)
        loss = criterion(logits, target)
        # rank_log(rank, logger, f"loss {loss}")
        loss.backward()
        # output.sum().backward()
        optimizer.step()
        rank_log(rank, logger, f"2D iter {i} complete")

    cleanup()
    rank_log(rank, logger, f"2D DP{dp_size}TP{tp_size} training successfully completed!")


# modal run src/train/demo/cp.py
@app.local_entrypoint()
def main():
    run.remote()
