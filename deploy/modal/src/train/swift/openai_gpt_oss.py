import os
import asyncio
import subprocess


import modal


app = modal.App("openai_gpt_oss_swift")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs")
    # https://github.com/modelscope/ms-swift/pull/5277
    .pip_install("ms-swift>=3.7.0")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        }
    )
)


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRAIN_OUTPUT_DIR = "/train_output"
train_out_vol = modal.Volume.from_name("train_output", create_if_missing=True)


with img.imports():
    import torch
    from modelscope.msdatasets import MsDataset
    from swift.llm import (
        get_model_tokenizer,
        load_dataset,
        get_template,
        EncodePreprocessor,
        InferEngine,
        InferRequest,
        PtEngine,
        RequestConfig,
    )
    from swift.utils import (
        get_logger,
        find_all_linears,
        get_model_parameter_info,
        plot_images,
        seed_everything,
    )

    from swift.tuners import Swift, LoraConfig
    from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    from functools import partial

    logger = get_logger()
    seed_everything(42)

    MODEL_PATH = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    output_dir = os.path.join(TRAIN_OUTPUT_DIR, f"{MODEL_PATH.split('/')[-1]}-swift")


@app.function(
    gpu=IMAGE_GPU,
    cpu=4.0,
    retries=1,
    image=img,
    #secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRAIN_OUTPUT_DIR: train_out_vol,
    },
    timeout=86400,  # default 300s
    max_containers=1,
)
def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    func(**kwargs)


def train(**kwargs):
    # Hyperparameters for training
    # model
    system = "You are a helpful assistant."

    # dataset
    dataset = [
        "AI-ModelScope/alpaca-gpt4-data-zh#500",
        "AI-ModelScope/alpaca-gpt4-data-en#500",
        "swift/self-cognition#500",
    ]  # dataset_id or dataset_path
    data_seed = 42
    max_length = 2048
    split_dataset_ratio = 0.01  # Split validation set
    num_proc = 4  # The number of processes for data loading.
    # The following two parameters are used to override the placeholders in the self-cognition dataset.
    model_name = ["小黄", "Xiao Huang"]  # The Chinese name and English name of the model
    model_author = ["魔搭", "ModelScope"]  # The Chinese name and English name of the model author

    # lora
    lora_rank = 8
    lora_alpha = 32

    # training_args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        router_aux_loss_coef=1e-3,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to=["tensorboard"],
        logging_first_step=True,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        metric_for_best_model="loss",
        save_total_limit=2,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
    )

    out_dir = os.path.abspath(os.path.expanduser(output_dir))
    logger.info(f"output_dir: {out_dir}")

    # Obtain the model and template, and add a trainable Lora layer on the model.
    model, tokenizer = get_model_tokenizer(model_path)
    logger.info(f"model_info: {model.model_info}")
    template = get_template(
        model.model_meta.template, tokenizer, default_system=system, max_length=max_length
    )
    template.set_mode("train")

    target_modules = find_all_linears(model)
    lora_config = LoraConfig(
        task_type="CAUSAL_LM", r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules
    )
    model = Swift.prepare_model(model, lora_config)
    logger.info(f"lora_config: {lora_config}")

    # Print model structure and trainable parameters.
    logger.info(f"model: {model}")
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f"model_parameter_info: {model_parameter_info}")

    # Download and load the dataset, split it into a training set and a validation set,
    # and encode the text data into tokens.
    train_dataset, val_dataset = load_dataset(
        dataset,
        split_dataset_ratio=split_dataset_ratio,
        num_proc=num_proc,
        model_name=model_name,
        model_author=model_author,
        seed=data_seed,
    )

    logger.info(f"train_dataset: {train_dataset}")
    logger.info(f"val_dataset: {val_dataset}")
    logger.info(f"train_dataset[0]: {train_dataset[0]}")

    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
    logger.info(f"encoded_train_dataset[0]: {train_dataset[0]}")

    # Print a sample
    template.print_inputs(train_dataset[0])

    # Get the trainer and start the training.
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    trainer.train()

    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f"last_model_checkpoint: {last_model_checkpoint}")

    # Visualize the training loss.
    # You can also use the TensorBoard visualization interface during training by entering
    # `tensorboard --logdir '{output_dir}/runs'` at the command line.
    images_dir = os.path.join(out_dir, "images")
    logger.info(f"images_dir: {images_dir}")
    plot_images(images_dir, training_args.logging_dir, ["train/loss"], 0.9)  # save images


def inference(**kwargs):
    lora_adapter_checkpoint = os.path.join(output_dir, "checkpoint-94")

    # 模型
    system = "You are a helpful assistant."
    # 'VllmEngine', 'LmdeployEngine', 'SglangEngine', 'PtEngine',
    infer_backend = kwargs.get("infer_backend", "pt")

    # 生成参数
    max_new_tokens = kwargs.get("max_new_tokens", 512)
    temperature = kwargs.get("temperature", 0.0)
    stream = kwargs.get("stream", True)

    engine: InferEngine = None
    if infer_backend == "pt":
        engine = PtEngine(model_path, adapters=[lora_adapter_checkpoint])
    else:
        raise ValueError("un support infer backend.")

    template = get_template(
        engine.model.model_meta.template, engine.tokenizer, default_system=system
    )
    # 这里对推理引擎的默认template进行修改，也可以在`engine.infer`时进行传入
    engine.default_template = template

    query_list = [
        "who are you?",
        "晚上睡不着觉怎么办？",
        "你是谁训练的？",
    ]

    def infer_stream(engine: InferEngine, infer_request: InferRequest):
        request_config = RequestConfig(
            max_tokens=max_new_tokens, temperature=temperature, stream=True
        )
        query = infer_request.messages[0]["content"]
        print(f"query: {query}")
        gen_list = engine.infer([infer_request], request_config)
        print(f"response:", end="", flush=True)
        for resp in gen_list[0]:
            if resp is None:
                continue
            print(resp.choices[0].delta.content, end="", flush=True)
        print()

    def infer(engine: InferEngine, infer_request: InferRequest):
        request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
        resp_list = engine.infer([infer_request], request_config)
        query = infer_request.messages[0]["content"]
        response = resp_list[0].choices[0].message.content
        print(f"query: {query}")
        print(f"response: {response}")

    infer_func = infer_stream if stream else infer
    for query in query_list:
        infer_func(engine, InferRequest(messages=[{"role": "user", "content": query}]))
        print("-" * 50)


"""
modal run src/download_models.py --repo-ids "openai/gpt-oss-20b"

IMAGE_GPU=L40s modal run src/train/swift/openai_gpt_oss.py --task train
IMAGE_GPU=H100 modal run src/train/swift/openai_gpt_oss.py --task train

IMAGE_GPU=L40s modal run src/train/swift/openai_gpt_oss.py --task inference 
IMAGE_GPU=L40s modal run src/train/swift/openai_gpt_oss.py --task inference --no-stream

# see: openai/gpt-oss-20b with Mxfp4HfQuantizer on T4 : https://colab.research.google.com/drive/1Gsgdydt4KgTm3S_Dbc_Gz08S2mQG4G8X?usp=sharing
IMAGE_GPU=T4 modal run src/train/swift/openai_gpt_oss.py --task inference 
IMAGE_GPU=T4 modal run src/train/swift/openai_gpt_oss.py --task inference --no-stream
"""


@app.local_entrypoint()
def main(
    task: str = "train",
    infer_backend: str = "pt",
    stream: bool = True,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
):
    print(task)
    tasks = {
        "train": train,
        "inference": inference,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
        infer_backend=infer_backend,
        stream=stream,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
