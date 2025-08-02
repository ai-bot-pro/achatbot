import os
import time
import logging
import asyncio
from typing import AsyncGenerator

try:
    import torch

    assert torch.__version__ >= "2.3.0", "torch version must be >= 2.3.0"
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
except ImportError as e:
    print("you need to `pip install achatbot[transformers]`")
    raise Exception(f"Missing module: {e}")

from src.common.session import Session
from src.modules.speech.asr.base import ASRBase
from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.modules.speech.help.audio_mock import generate_random_sine


class WhisperTransformersTorchCompileAsr(ASRBase):
    """
    - https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    - https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html (need pytorch>2.4)
    - https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
    """

    TAG = "whisper_transformers_torch_compile_asr"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        torch.set_float32_matmul_precision("high")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.args.model_name_or_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        ).to(self.device)

        # Enable static cache and compile the forward pass
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = 256
        model.config.ignore_logger_methods = True
        # cuda graphs compile forward
        model.forward = torch.compile(
            model.forward,
            mode=self.args.torch_compile_mode,
            fullgraph=True,
            # options={"triton.cudagraphs": self.args.triton_cudagraphs},
        )
        self.model = model
        self.processor = AutoProcessor.from_pretrained(self.args.model_name_or_path)

        # warmup
        for i in range(self.args.warmup_steps):
            i == 0 and logging.info(f"Start Warmuping {self.args.warmup_steps} steps")
            audio_np = generate_random_sine(5, self.args.sample_rate)
            input_ids = self.processor(
                audio_np,
                sampling_rate=self.args.sample_rate,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
            ).to(self.device, dtype=self.torch_dtype)

            with sdpa_kernel(SDPBackend.MATH):
                start = time.perf_counter()
                self.model.generate(
                    **input_ids,
                    min_new_tokens=256,
                    max_new_tokens=256,
                )
                logging.info(f"Warmup {i} step took {time.perf_counter() - start:.2f} seconds")


    def gen(self, input_ids: dict, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            pred_ids = self.model.generate(**input_ids, **kwargs)
            return pred_ids

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        res = await self.transcribe(session)
        yield res["text"]

    async def transcribe(self, session: Session) -> dict:
        input_ids = self.processor(
            self.asr_audio.copy(),
            sampling_rate=self.args.sample_rate,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
        ).to(self.device, dtype=self.torch_dtype)

        pred_ids = await asyncio.to_thread(self.gen, input_ids, task="transcribe")
        pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        res = {
            "language": self.args.language,
            "language_probability": None,
            "text": pred_text[0].strip(),
            "words": [],
        }
        return res


class WhisperTransformersPipelineTorchCompileAsr(ASRBase):
    """
    - https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    - https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html (need pytorch>2.4)
    - https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
    """

    TAG = "whisper_transformers_pipeline_torch_compile_asr"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        torch.set_float32_matmul_precision("high")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.args.model_name_or_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        ).to(self.device)

        # Enable static cache and compile the forward pass
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = 256
        model.config.ignore_logger_methods = True
        # cuda graphs compile forward
        model.forward = torch.compile(
            model.forward, mode=self.args.torch_compile_mode, fullgraph=True
        )
        self.model = model
        self.processor = AutoProcessor.from_pretrained(self.args.model_name_or_path)

        #  ⚠️Note:
        # torch.compile is currently not compatible with the Chunked long-form algorithm or Flash Attention 2
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            # chunk_length_s=30,
            # batch_size=self.args.batch_size,
        )

        # warmup
        for i in range(self.args.warmup_steps):
            i == 0 and logging.info(f"Start Warmuping {self.args.warmup_steps} steps")
            audio_np = generate_random_sine(5, self.args.sample_rate)

            with sdpa_kernel(SDPBackend.MATH):
                start = time.perf_counter()
                self.pipe(
                    audio_np,
                    generate_kwargs={
                        "min_new_tokens": 256,
                        "max_new_tokens": 256,
                        "task": "transcribe",
                    },
                    return_timestamps=True,
                )
                logging.info(f"Warmup {i} step took {time.perf_counter() - start:.2f} seconds")

    def pipe_gen(self, audio_np, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return self.pipe(audio_np, **kwargs)

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        outputs = self.pipe_gen(
            self.asr_audio if isinstance(self.asr_audio, str) else self.asr_audio.copy(),
            generate_kwargs={
                "language": self.args.language,
                "task": "transcribe",
            },
            chunk_length_s=30,
            batch_size=1,
            return_timestamps=True,
        )
        for item in outputs["chunks"]:
            yield item["text"]

    async def transcribe(self, session: Session) -> dict:
        outputs = self.pipe_gen(
            self.asr_audio if isinstance(self.asr_audio, str) else self.asr_audio.copy(),
            generate_kwargs={
                "language": self.args.language,
                "task": "transcribe",
            },
            chunk_length_s=30,
            batch_size=1,
            return_timestamps=True,
        )
        res = {
            "language": self.args.language,
            "language_probability": None,
            "text": outputs["text"].strip(),
            "words": [
                {"text": item["text"], "start": item["timestamp"][0], "end": item["timestamp"][1]}
                for item in outputs["chunks"]
            ],
        }
        return res
