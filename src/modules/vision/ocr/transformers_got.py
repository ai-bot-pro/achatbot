import logging
import os
import sys
from threading import Thread
import time

import torch
from PIL import Image

from src.core.llm.base import BaseLLM
from src.common.session import Session
from src.common.interface import IVisionOCR
from src.types.vision.ocr.transformers_got import TransformersGoTOCRArgs

try:
    from qwen_vl_utils import fetch_image
    from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer

    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../GOTOCR2"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/GOTOCR2"))
    from deps.GOTOCR2.GOT.utils.utils import disable_torch_init, KeywordsStoppingCriteria
    from deps.GOTOCR2.GOT.utils.conversation import conv_templates, SeparatorStyle
    from deps.GOTOCR2.GOT.model.plug.blip_process import BlipImageEvalProcessor
    from deps.GOTOCR2.GOT.model import GOTQwenForCausalLM
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use GOT OCR2.0 language models, you need to `pip install achatbot[vision_transformers_got_ocr]`,"
        "use awq model need to `pip install achatbot[vision_transformers_got_ocr]`"
    )
    raise Exception(f"Missing module: {e}")

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"

DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"


class TransformersGOTOCRLM(BaseLLM, IVisionOCR):
    """
    the General OCR Theory (GOT), ViTDet(vision encoder) + qwen2 0.5B
    the ViTDet backbone available at:
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py

    """

    TAG = "vision_transformers_got_ocr"

    def __init__(self, **args) -> None:
        self.args = TransformersGoTOCRArgs(**args)

        if hasattr(torch, self.args.lm_torch_dtype):
            self.torch_dtype = getattr(torch, self.args.lm_torch_dtype)
        else:
            raise Exception(f"torch unsupport dtype: {self.args.lm_torch_dtype}")

        disable_torch_init()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True
        )

        if self.args.lm_device_map:
            # self._model = AutoModel.from_pretrained(
            self._model = GOTQwenForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.torch_dtype,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation=self.args.lm_attn_impl,
                use_safetensors=True,
                pad_token_id=self._tokenizer.eos_token_id,
                trust_remote_code=True,
            ).eval()
        else:
            # self._model = AutoModel.from_pretrained(
            self._model = (
                GOTQwenForCausalLM.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=self.torch_dtype,
                    attn_implementation=self.args.lm_attn_impl,
                    use_safetensors=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.args.lm_device)
            )
        logging.debug(f"GOT model:{self._model}, device: {self._model.device}")

        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self._image: Image.Image = None

        self.warmup()

    def count_tokens(self, text: str | bytes) -> int:
        return len(self._tokenizer.encode(text)) if self._tokenizer else 0

    def warmup(self):
        pass

    def set_image(self, session: Session):
        if "ocr_img" not in session.ctx.state:
            logging.warning("no ocr_img in session.ctx.state")
            return False

        ocr_image = session.ctx.state["ocr_img"]
        if isinstance(ocr_image, dict) and all(
            isinstance(k, str) and isinstance(v, (str, Image.Image)) for k, v in ocr_image.items()
        ):
            self._image = fetch_image(ocr_image)
        elif isinstance(ocr_image, Image.Image):
            self._image = ocr_image
        else:
            raise ValueError(
                f"Unrecognized image input: PIL.Image or dict key:image/image_url, val:support local path, http url, base64 and PIL.Image, got {ocr_image}"
            )

        logging.debug(f"ocr image: {self._image}")
        return True

    def _generate(self, session: Session):
        """
        use huggingface transformers AutoModel load model to inference
        """
        if self.set_image(session) is False:
            yield None
            return

        generation_kwargs = dict(
            ocr_type=self.args.ocr_type,
            ocr_box=self.args.ocr_box,
            ocr_color=self.args.ocr_color,
            stream_flag=True,
            gradio_input=True,
            streamer=self._streamer,
        )
        thread = Thread(
            target=self._model.chat, args=(self._tokenizer, self._image), kwargs=generation_kwargs
        )
        thread.start()

        for new_text in self._streamer:
            yield new_text

    def generate(self, session: Session):
        res = ""
        if self.args.lm_stream is False:
            for text in self.stream_infer(session):
                if text is None:
                    continue
                res += text
        else:
            res = ""
            for text in self.stream_infer(session):
                if text is None:
                    continue
                res += text
                pos = self._have_special_char(res)
                if pos > -1:
                    yield res[: pos + 1]
                    res = res[pos + 1 :]
        if len(res) > 0:
            yield res + "."

    def stream_infer(self, session: Session):
        """
        use GOT model lib to inference with streamer
        """
        if self.set_image(session) is False:
            yield None
            return

        if self.args.ocr_type == "format":
            qs = "OCR with format: "
        else:
            qs = "OCR: "

        w, h = self._image.size
        if self.args.ocr_box:
            bbox = eval(self.args.ocr_box)
            if len(bbox) == 2:
                bbox[0] = int(bbox[0] / w * 1000)
                bbox[1] = int(bbox[1] / h * 1000)
            if len(bbox) == 4:
                bbox[0] = int(bbox[0] / w * 1000)
                bbox[1] = int(bbox[1] / h * 1000)
                bbox[2] = int(bbox[2] / w * 1000)
                bbox[3] = int(bbox[3] / h * 1000)
            if self.args.ocr_type == "format":
                qs = str(bbox) + " " + "OCR with format: "
            else:
                qs = str(bbox) + " " + "OCR: "

        if self.args.ocr_color:
            if self.args.ocr_type == "format":
                qs = "[" + self.args.ocr_color + "]" + " " + "OCR with format: "
            else:
                qs = "[" + self.args.ocr_color + "]" + " " + "OCR: "

        use_im_start_end = True
        image_token_len = 256
        if use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        #!TODO: need config system prompt
        # self.args.conv_mode = "mpt"
        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        logging.debug(f"ocr prompt: {prompt}")

        # text input tokens
        inputs = self._tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).to(self._model.device)

        # ocr image input
        image_processor_high = BlipImageEvalProcessor(image_size=1024)
        image_tensor = image_processor_high(self._image)
        image = image_tensor.unsqueeze(0).half().to(self._model.device)

        # stoping criteria (end/stop tokens)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self._tokenizer, input_ids)

        device_type = "cuda" if "cuda" in str(self._model.device) else "cpu"
        logging.debug(
            f"inference device_type: {device_type}, image shape:{image.shape}, image dtype:{image.dtype}"
        )

        def run():
            with torch.autocast(device_type, dtype=self.torch_dtype):
                self._model.generate(
                    input_ids,
                    images=[(None, image)],
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=20,
                    streamer=self._streamer,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping_criteria],
                )

        thread = Thread(target=run)
        thread.start()

        for new_text in self._streamer:
            yield new_text
