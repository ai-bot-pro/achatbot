import os
import logging
from typing import AsyncGenerator, Literal

try:
    import torch
    from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
except ModuleNotFoundError as e:
    logging.error(
        "In order to use diffusers, you need to `pip install achatbot[diffusers,bitsandbytes]`."
        "Also, set `HF_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

from PIL import Image
from apipeline.frames.data_frames import Frame, ImageRawFrame
from apipeline.frames.sys_frames import ErrorFrame

from src.processors.image.base import ImageGenProcessor


# https://huggingface.co/docs/api-inference/tasks/text-to-image


class HFStableDiffusionImageGenProcessor(ImageGenProcessor):
    def __init__(
        self,
        *,
        model: str = "stabilityai/stable-diffusion-3.5-large",
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        guidance_scale: float = 7.5,  # like temperature
        is_quantizing: bool = False,
        negative_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality",
        device: Literal["auto", "cpu", "cuda"] = "auto",
    ):
        super().__init__()
        self._width = width
        self._height = height
        self._steps = steps
        self._guidance_scale = guidance_scale
        self._negative_prompt = negative_prompt
        if is_quantizing:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                model,
                subfolder="transformer",
                quantization_config=nf4_config,
                token=os.environ.get("HF_API_KEY"),
                torch_dtype=torch.bfloat16,
            )
            self._pipe = StableDiffusion3Pipeline.from_pretrained(
                model,
                token=os.environ.get("HF_API_KEY"),
                torch_dtype=torch.bfloat16,
                transformer=model_nf4,
            )
        else:
            self._pipe = StableDiffusion3Pipeline.from_pretrained(
                model,
                token=os.environ.get("HF_API_KEY"),
                torch_dtype=torch.bfloat16,
            )

        if device == "auto":
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(device)
        # self._pipe.enable_xformers_memory_efficient_attention()
        logging.debug(f"sd pipeline: {self._pipe}, device: {self._pipe.device}")

    def set_aiohttp_session(self, session):
        pass

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating image from prompt: {prompt}")

        image = self._pipe(
            prompt,
            num_inference_steps=self._steps,
            guidance_scale=self._guidance_scale,
            height=self._height,
            width=self._width,
            negative_prompt=self._negative_prompt,
            generator=torch.Generator(device=self._pipe.device).manual_seed(2),
        ).images[0]

        if not image:
            yield ErrorFrame("Image generation failed")
            return

        image = image.convert("RGB")
        # if image.size != (self._width, self._height):
        #    image = image.resize((self._width, self._height))
        frame = ImageRawFrame(
            image=image.tobytes(),
            size=image.size,
            format=image.format if image.format else "JPEG",
            mode=image.mode,
        )
        yield frame
