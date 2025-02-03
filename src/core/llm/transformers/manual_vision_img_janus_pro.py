import io
import logging
import os
import sys
from threading import Thread
from dotenv import load_dotenv

from PIL import Image

from src.common.session import Session
from src.common.utils.helper import get_device
from src.core.llm.transformers.base import TransformersBaseLLM
from src.types.llm.transformers import TransformersLMArgs
from src.common.random import set_all_random_seed

load_dotenv(override=True)

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../Janus"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/Janus"))
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig, TextIteratorStreamer
    from deps.Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
    from deps.Janus.janus.utils.io import load_pil_images
except ModuleNotFoundError as e:
    logging.error(
        "In order to use DeepSeek janus, you need to `pip install achatbot[llm_transformers_manual_vision_img_janus]`."
    )
    raise Exception(f"Missing module: {e}")


class TransformersManualJanusPro(TransformersBaseLLM):
    r"""
    Multimodal Understanding + Text-to-Image Generation
    https://github.com/deepseek-ai/Janus

    vl_chat_processor.tokenizer + AR LM model + gen_vision_model
    """

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)

        # https://huggingface.co/deepseek-ai/Janus-Pro-1B/blob/main/config.json
        # https://huggingface.co/deepseek-ai/Janus-Pro-7B/blob/main/config.json
        config = AutoConfig.from_pretrained(self.args.lm_model_name_or_path)
        language_config = config.language_config
        language_config._attn_implementation = "eager"
        self._model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.args.lm_model_name_or_path,
            language_config=language_config,
            trust_remote_code=True,
        )
        self._model = self._model.to(self.args.lm_device).eval()

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            self.args.lm_model_name_or_path
        )
        self._tokenizer = self.vl_chat_processor.tokenizer

        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        self.warmup()


class TransformersManualVisionJanusPro(TransformersManualJanusPro):
    r"""
    Multimodal Understanding
    https://github.com/deepseek-ai/Janus

    vl_chat_processor.tokenizer.encode + AR LM model + vl_chat_processor.tokenizer.decode
    """

    TAG = "llm_transformers_manual_vision_janus"

    def warmup(self):
        dummy_input_text = self.args.warnup_prompt
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{dummy_input_text}",
            },
            {"role": "Assistant", "content": ""},
        ]

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=None, force_batchify=True
        ).to(
            self.args.lm_device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        logging.debug(f"prepare_inputs: {prepare_inputs}")

        # input embeddings
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)
        # AR lm generate with streamer
        warmup_gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            streamer=self._streamer,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self._tokenizer.eos_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            do_sample=False if self.args.lm_gen_temperature == 0 else True,
            use_cache=True,
            temperature=self.args.lm_gen_temperature,
            top_p=self.args.lm_gen_top_p,
        )

        self._warmup(
            target=self._model.language_model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=self._streamer,
        )

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        if "cuda" in str(self._model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)
        question = session.ctx.state["prompt"]
        image_data = session.ctx.state["image_data"]  # bytes

        # conversation
        message = {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            # "images": [image_data],
        }
        self._chat_history.append(message)
        chat_history = self._chat_history.to_list()
        logging.debug(f"chat_history:{chat_history}")
        conversation = chat_history + [{"role": "Assistant", "content": ""}]

        # inputs
        pil_images = [Image.open(io.BytesIO(image_data))]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(
            self.args.lm_device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        logging.debug(f"prepare_inputs: {prepare_inputs}")

        # input embeddings
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

        # AR lm generate with streamer
        generation_kwargs = dict(
            inputs_embeds=inputs_embeds,
            streamer=self._streamer,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self._tokenizer.eos_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            do_sample=False if self.args.lm_gen_temperature == 0 else True,
            use_cache=True,
            temperature=self.args.lm_gen_temperature,
            top_p=self.args.lm_gen_top_p,
        )
        thread = Thread(target=self._model.language_model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in self._streamer:
            generated_text += new_text
            yield new_text
        self._chat_history.append({"role": "Assistant", "content": generated_text})


class TransformersManualGenImageJanusPro(TransformersManualJanusPro):
    r"""
    Text-to-Image Generation
    https://github.com/deepseek-ai/Janus

    vl_chat_processor.tokenizer.encode + AR LM model + gen_vision_model.decode
    """

    TAG = "llm_transformers_manual_image_janus"

    def warmup(self):
        pass

    @torch.inference_mode()
    def generate(self, session: Session):
        pass
