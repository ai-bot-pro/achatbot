from dataclasses import dataclass, field


@dataclass
class TransformersGoTOCRArgs:
    r"""
    just use transformers Got OCR2.0
    """

    lm_model_name_or_path: str = field(
        # see: https://huggingface.co/stepfun-ai/GOT-OCR2_0/discussions/26
        default="weege007/GOT-OCR2_0",  # from stepfun-ai/GOT-OCR2_0, change modeling_GOT.py for streamer
        metadata={
            "help": "The pretrained language model to use. Default is 'weege007/GOT-OCR2_0'."
        },
    )
    lm_device_map: str | dict | None = field(
        default=None,
        metadata={
            "help": "The device map for multi cpu/gpu. use 'cuda' for GPU acceleration, 'mps','cpu','auto', dict, None. default None"
        },
    )
    lm_device: str = field(
        default="cpu",
        metadata={
            "help": "The device for single cpu/mps/gpu. use 'cuda' for GPU acceleration, 'mps'(apple), 'cpu'. default cpu"
        },
    )
    lm_torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision). default bfloat16"
        },
    )
    # https://huggingface.co/docs/transformers/perf_infer_gpu_one
    lm_attn_impl: str | None = field(
        default=None,
        metadata={
            "help": "The attention implementation to use. One of 'sdpa', 'flash_attention_2', default no attention implementation."
        },
    )
    warnup_prompt: str = field(
        default="Repeat the word 'weedge niu bi'.",
        metadata={"help": "warnup llm generate prompt. Default is 'weedge niu bi'."},
    )
    warmup_steps: int = field(
        default=2,
        metadata={"help": "The number of steps to run the warmup prompt. Default is 2."},
    )
    lm_stream: bool = field(
        default=True,
        metadata={
            "help": "Whether to use streaming; set this to True for streaming output. Default is True."
        },
    )
    model_type: str = field(
        default="chat",
        metadata={"help": "Model type, chat, chat_crop. Default is chat."},
    )
    ocr_type: str = field(
        default="ocr",
        metadata={"help": "OCR type, ocr,format. Default is ocr."},
    )
    ocr_box: str = field(
        default="",
        metadata={
            "help": "ocr_box bbox, e.g.:'[x1,y1]' or '[x1,y1,x2,y2]' . Default is empty str."
        },
    )
    ocr_color: str = field(
        default="",
        metadata={"help": "ocr_color, e.g.:'GREEN'. Default is empty str."},
    )
    conv_mode: str = field(
        default="mpt",
        metadata={"help": "conversation mode, e.g.:'mpt'. Default is 'mpt'."},
    )
    render: bool = field(
        default=False,
        metadata={"help": "Whether to render to file, Default is false."},
    )
    save_render_file: str = field(
        default="",
        metadata={"help": "when render is true, to save file path, Default is empty str."},
    )
