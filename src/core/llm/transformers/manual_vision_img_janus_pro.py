import io
import logging
import os
import sys
from threading import Thread
from dotenv import load_dotenv

from PIL import Image
import numpy as np

from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.common.utils.helper import get_device, print_model_params
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
    base class for Multimodal Understanding + Text-to-Image Generation
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
        print_model_params(self._model, self.TAG)

        self._model = self._model.to(
            self.args.lm_device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        ).eval()

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            self.args.lm_model_name_or_path
        )
        self._tokenizer = self.vl_chat_processor.tokenizer

        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        self._chat_history = ChatHistory(self.args.chat_history_size)
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

        dummy_pil_images = [Image.new("RGB", (100, 100), color="white")]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=dummy_pil_images, force_batchify=True
        ).to(
            self._model.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        # logging.debug(f"prepare_inputs: {prepare_inputs}")

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
        logging.debug(f"kwargs: {kwargs}")
        if "cuda" in str(self._model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        assert isinstance(session.ctx.state["prompt"], list)
        assert len(session.ctx.state["prompt"]) > 1
        question = session.ctx.state["prompt"][-1]
        pil_images = session.ctx.state["prompt"][:-1]

        lm_gen_max_new_tokens = kwargs.get("lm_gen_max_new_tokens", self.args.lm_gen_max_new_tokens)
        lm_gen_temperature = kwargs.get("lm_gen_temperature", self.args.lm_gen_temperature)
        lm_gen_top_p = kwargs.get("lm_gen_top_p", self.args.lm_gen_top_p)

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
        # pil_images = [Image.open(io.BytesIO(image_data))]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(
            self._model.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        # logging.debug(f"prepare_inputs: {prepare_inputs}")

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
            max_new_tokens=lm_gen_max_new_tokens,
            do_sample=False if lm_gen_temperature == 0 else True,
            use_cache=True,
            temperature=lm_gen_temperature,
            top_p=lm_gen_top_p,
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
    Text-to-Image Generation image(384*384)
    https://github.com/deepseek-ai/Janus

    vl_chat_processor.tokenizer.encode + AR LM model + gen_vision_model.decode
    """

    TAG = "llm_transformers_manual_image_janus"

    def warmup(self):
        pass

    def unpack(self, dec, width, height, parallel_size=1):
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        return visual_img

    @torch.no_grad()
    def _generate(
        self,
        input_ids,
        width,
        height,
        temperature: float = 1,
        parallel_size: int = 5,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        patch_size: int = 16,
    ):
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(
            self._model.device
        )
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id
        inputs_embeds = self._model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros(
            (parallel_size, image_token_num_per_image), dtype=torch.int
        ).to(self._model.device)

        pkv = None
        for i in range(image_token_num_per_image):
            outputs = self._model.language_model.model(
                inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv
            )
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = self._model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
            ).view(-1)
            img_embeds = self._model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        patches = self._model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, width // patch_size, height // patch_size],
        )

        return generated_tokens.to(dtype=torch.int), patches

    @torch.no_grad()
    def _gen_image(self, prompt, guidance, parallel_size=1, gen_width=1024, gen_height=1024):
        width = 384
        height = 384

        messages = [
            {"role": "User", "content": prompt},
            {"role": "Assistant", "content": ""},
        ]
        text = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        text = text + self.vl_chat_processor.image_start_tag
        input_ids = torch.LongTensor(self._tokenizer.encode(text))
        _, patches = self._generate(
            input_ids,
            width // 16 * 16,
            height // 16 * 16,
            cfg_weight=guidance,
            parallel_size=parallel_size,
        )
        images = self.unpack(
            patches,
            width // 16 * 16,
            height // 16 * 16,
            parallel_size=parallel_size,
        )

        return [
            Image.fromarray(images[i]).resize((gen_width, gen_height), Image.LANCZOS)
            for i in range(parallel_size)
        ]

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        logging.debug(f"kwargs: {kwargs}")
        prompt = session.ctx.state["prompt"]
        if "cuda" in str(self._model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        guidance = kwargs.get("guidance", 5.0)
        parallel_size = kwargs.get("parallel_size", 1)
        gen_width = kwargs.get("gen_width", 1024)
        gen_height = kwargs.get("gen_height", 1024)
        images = self._gen_image(
            prompt,
            guidance,
            parallel_size=parallel_size,
            gen_width=gen_width,
            gen_height=gen_height,
        )
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            yield buf.read()


class TransformersManualVisionGenImageJanusPro(
    TransformersManualGenImageJanusPro, TransformersManualVisionJanusPro
):
    r"""
    Multimodal Understanding + Text-to-Image Generation
    https://github.com/deepseek-ai/Janus

    vl_chat_processor.tokenizer + AR LM model + gen_vision_model
    """

    def generate(self, session: Session, **kwargs):
        r"""
        According to Python's MRO (Method Resolution Order) rules:
        In multiple inheritance, Python searches parent classes from left to right
        Using the generate method from the first parent class, we need to define the generate method
        to distinguish based on parameters
        """
        if isinstance(session.ctx.state.get("prompt"), list):
            return TransformersManualVisionJanusPro.generate(self, session, **kwargs)
        else:
            return TransformersManualGenImageJanusPro.generate(self, session, **kwargs)
