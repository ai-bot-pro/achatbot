import io
import logging
from threading import Thread
from PIL import Image
from time import perf_counter

try:
    from transformers import AutoTokenizer, TextIteratorStreamer, AutoModel
    import torch

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Deepseek-OCR, you need to `pip install achatbot[llm_transformers_manual_vision_deepseek_ocr]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.utils.helper import get_device, print_model_params
from src.common.interface import IVisionOCR
from src.common.random import set_all_random_seed
from src.common.session import Session
from src.types.speech.language import TO_LLM_LANGUAGE
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


class TransformersManualVisionDeepSeekOCR(TransformersBaseLLM, IVisionOCR):
    TAG = "llm_transformers_manual_vision_deepseek_ocr"

    def __init__(self, tokenizer=None, **args) -> None:
        self.base_size = args.pop("ocr_base_size", 1024)
        self.image_size = args.pop("ocr_image_size", 640)
        self.crop_mode = args.pop("ocr_crop_mode", True)
        # Tiny: base_size = 512, image_size = 512, crop_mode = False
        # Small: base_size = 640, image_size = 640, crop_mode = False
        # Base: base_size = 1024, image_size = 1024, crop_mode = False
        # Large: base_size = 1280, image_size = 1280, crop_mode = False
        # Gundam: base_size = 1024, image_size = 640, crop_mode = True # default

        self.prompt = args.pop(
            "ocr_prompt", "<image>\n<|grounding|>Convert the document to markdown. "
        )
        # document: <image>\n<|grounding|>Convert the document to markdown.
        # other image: <image>\n<|grounding|>OCR this image.
        # without layouts: <image>\nFree OCR.
        # figures in document: <image>\nParse the figure.
        # general: <image>\nDescribe this image in detail.
        # rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.

        self.args = TransformersLMArgs(**args)
        gpu_prop = torch.cuda.get_device_properties("cuda") if torch.cuda.is_available() else None

        if self.args.lm_device_map:
            self._model = AutoModel.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=torch.bfloat16,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation="flash_attention_2"
                if gpu_prop and gpu_prop.major >= 8
                else None,
                trust_remote_code=True,
            ).eval()
        else:
            self._model = (
                AutoModel.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                    if gpu_prop and gpu_prop.major >= 8
                    else None,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.args.lm_device)
            )

        logging.info(f"TransformersLMArgs: {self.args}")
        print_model_params(self._model, self.TAG)
        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, use_fast=True
        )
        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0:
            return

        # create dummy image file
        dummy_img = Image.new("RGB", (640, 640), color="white")
        ioBuff = io.BytesIO()
        dummy_img.save(ioBuff, format="PNG")
        ioBuff.seek(0)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=False
        )

        warmup_gen_kwargs = dict(
            tokenizer=self._tokenizer,
            prompt=self.prompt,
            image_file=ioBuff,
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            save_results=False,
            test_compress=False,
            eval_mode=False,
            verbose=False,
            streamer=streamer,
        )

        self._warmup(
            target=self._model.infer,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    def set_task_prompt(self, prompt: str):
        self.prompt = prompt

    @torch.inference_mode()
    async def async_generate(self, session: Session, **kwargs):
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        ocr_img = session.ctx.state["ocr_img"]
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        generation_kwargs = dict(
            tokenizer=self._tokenizer,
            prompt=self.prompt,
            image_file=ocr_img,
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            save_results=False,
            test_compress=False,
            eval_mode=False,
            verbose=False,
            streamer=streamer,
        )
        thread = Thread(target=self._model.infer, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        start = perf_counter()
        times = []
        is_ref_det = False
        sentence = ""
        # TODO: extract detect image to storage with s3 (use callback)
        for new_text in streamer:
            times.append(perf_counter() - start)
            generated_text += new_text
            if "<|ref|>" in new_text:
                is_ref_det = True

            if "<|/det|>" in new_text:
                is_ref_det = False
                new_text = new_text.split("<|/det|>")[1]

            if "<｜end▁of▁sentence｜>" in new_text:
                if "<|/ref|>" not in new_text:
                    new_text = new_text.split("<｜end▁of▁sentence｜>")[0]
            if "<|end▁of▁sentence|>" in new_text:
                if "<|/ref|>" not in new_text:
                    new_text = new_text.split("<|end▁of▁sentence|>")[0]

            if is_ref_det is False:
                sentence += new_text
                pos = self._have_special_char(sentence)
                if pos > -1:
                    yield sentence[: pos + 1]
                    sentence = sentence[pos + 1 :]
            start = perf_counter()
        if len(sentence) > 0:
            yield sentence + "."
        if times:
            logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")
        torch.cuda.empty_cache()
