import copy
import logging
from threading import Thread
from PIL import Image
from time import perf_counter

try:
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        TextIteratorStreamer,
    )

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use phi model, you need to `pip install achatbot[llm_transformers_manual_vision_phi]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.utils.helper import get_device, print_model_params
from src.common.random import set_all_random_seed
from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.types.speech.language import TO_LLM_LANGUAGE
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


class TransformersManualVisionSpeechPhiLM(TransformersBaseLLM):
    TAG = "llm_transformers_manual_phi4_vision_speech"

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)
        gpu_prop = torch.cuda.get_device_properties("cuda")

        if self.args.lm_device_map:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=torch.bfloat16,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
                trust_remote_code=True,
            ).eval()
        else:
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.args.lm_device)
            )

        logging.info(f"TransformersLMArgs: {self.args}")
        print_model_params(self._model, self.TAG)
        self._tokenizer = AutoProcessor.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True, use_fast=True
        )

        self._chat_history = ChatHistory(self.args.chat_history_size)
        if self.args.init_chat_role and self.args.init_chat_prompt:
            self._chat_history.init(
                {
                    "role": self.args.init_chat_role,
                    "content": self.args.init_chat_prompt,
                }
            )

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        dummy_input_text = self.args.warnup_prompt
        dummy_pil_image = Image.new("RGB", (100, 100), color="white")
        dummy_msgs = [
            {
                "role": self.args.user_role,
                "content": [
                    {"type": "text", "text": dummy_input_text},
                    {"type": "image", "image": dummy_pil_image},
                ],
            }
        ]
        inputs = self.get_inputs(dummy_msgs)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        warmup_gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            do_sample=True if self.args.lm_gen_temperature > 0 else False,
            temperature=self.args.lm_gen_temperature,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=128,
        )

        self._warmup(
            target=self._model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    def cover_chat(self, chat: list):
        new_chat = []
        audio_cn = 0
        image_cn = 0
        for item in chat:
            new_chat.append(copy.deepcopy(item))
            tmp_text = ""
            tmp_content = ""
            sub_audio_cn = 0
            sub_image_cn = 0
            if "content" in item and isinstance(item["content"], list):
                for c_item in item["content"]:
                    assert isinstance(c_item, dict)
                    if "text" in c_item:
                        tmp_text = c_item["text"]
                    if "image" in c_item:
                        sub_image_cn += 1
                    if "audio" in c_item:
                        sub_audio_cn += 1
                image_cn += sub_image_cn
                audio_cn += sub_audio_cn
                for i in range(image_cn - sub_image_cn, image_cn):
                    tmp_content += f"<|image_{i+1}|>"
                for i in range(audio_cn - sub_audio_cn, audio_cn):
                    tmp_content += f"<|audio_{i+1}|>"
                if tmp_text:
                    tmp_content += tmp_text
                new_chat[-1]["content"] = tmp_content

        return new_chat

    # https://huggingface.co/microsoft/Phi-4-multimodal-instruct#input-formats
    def get_inputs(self, chat: list):
        audios, images, _ = process_mm_info(chat, use_audio_in_video=False)
        {
            logging.debug(f"audios[{i}]: {item.shape}") for i, item in enumerate(audios)
        } if audios else logging.debug(audios)
        {
            logging.debug(f"images[{i}]: {item}") for i, item in enumerate(images)
        } if images else logging.debug(images)

        # text promt
        chat = self.cover_chat(chat)
        # print(chat)
        prompt = self._tokenizer.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
        if prompt.endswith("<|endoftext|>"):
            prompt = prompt.rstrip("<|endoftext|>")
        logging.info(f"{prompt=}")

        if audios is not None:
            new_audios = []
            for audio in audios:
                new_audios.append((audio, 16000))
            audios = new_audios if new_audios else None
        # tokenize (tokens -> token_ids)
        inputs = self._tokenizer(text=prompt, images=images, audios=audios, return_tensors="pt").to(
            self._model.device, dtype=torch.bfloat16
        )
        for key, value in inputs.items():
            if value is not None:
                logging.info(f"{key}: {value.shape=}")
                pass
        return inputs

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = session.ctx.state["prompt"]
        # logging.info(prompt)
        assert len(prompt) > 0
        assert isinstance(prompt[0], dict)
        message = {"role": self.args.user_role, "content": prompt}
        self._chat_history.append(message)
        chat_history = self._chat_history.to_list()
        logging.debug(f"chat_history:{chat_history}")

        inputs = self.get_inputs(chat_history)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            do_sample=True
            if kwargs.get("temperature", self.args.lm_gen_temperature) > 0
            else False,
            temperature=kwargs.get("temperature", self.args.lm_gen_temperature),
            top_k=kwargs.get("top_k", self.args.lm_gen_top_k),
            top_p=kwargs.get("top_p", self.args.lm_gen_top_p),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.args.lm_gen_repetition_penalty
            ),
            min_new_tokens=kwargs.get("min_new_tokens", self.args.lm_gen_min_new_tokens),
            max_new_tokens=kwargs.get("max_new_tokens", self.args.lm_gen_max_new_tokens),
            use_cache=True,
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        start = perf_counter()
        times = []
        with torch.inference_mode():
            for new_text in streamer:
                times.append(perf_counter() - start)
                generated_text += new_text
                yield {"text": new_text.replace("*", "")}
                start = perf_counter()
        logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")
        self._chat_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": generated_text}]}
        )
        torch.cuda.empty_cache()


class TransformersManualAudioPhiLM(TransformersManualVisionSpeechPhiLM):
    """
    audio understanding

    - speech -> text
    """

    TAG = [
        "llm_transformers_manual_phi4_audio_asr",
        "llm_transformers_manual_phi4_audio_translation",
    ]

    def generate(self, session: Session, **kwargs):
        for item in super().generate(session, **kwargs):
            text = item.pop("text", "")
            if text == "":
                continue
            yield text


class TransformersManualVisionPhiLM(TransformersManualVisionSpeechPhiLM):
    """
    vision understanding

    - vision(image) -> text
    """

    TAG = "llm_transformers_manual_phi4_vision"

    def generate(self, session: Session, **kwargs):
        for item in super().generate(session, **kwargs):
            yield item["text"]


class TransformersManualAudioChatPhiLM(TransformersManualVisionSpeechPhiLM):
    """
    audio chat

    - speech -> text
    """

    TAG = [
        "llm_transformers_manual_phi4_audio_chat",
    ]
