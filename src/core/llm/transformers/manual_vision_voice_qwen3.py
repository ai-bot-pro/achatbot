import logging
from time import perf_counter
import time
import traceback
from threading import Thread
from typing import Dict
from functools import lru_cache

import numpy as np

try:
    import torch

    from qwen_omni_utils import process_mm_info
    from transformers import (
        AutoConfig,
        TextIteratorStreamer,
        Qwen3OmniMoeProcessor,
    )
    from src.core.llm.transformers.models.qwen3_omni import (
        Qwen3OmniMoeForConditionalGenerationStreaming,
    )

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Qwen3Omni, you need to `pip install achatbot[llm_transformers_manual_vision_voice_qwen]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.utils.thread_safe import ThreadSafeDict
from src.common.utils.helper import get_device
from src.common.random import set_all_random_seed
from src.common.session import Session
from src.types.omni.qwen3_vision_voice import (
    Qwen3TransformersVisionVoiceLMArgs,
    Qwen3OmniCode2WavArgs,
)
from src.types.llm.transformers import TransformersLMArgs
from src.types.speech.language import TO_LLM_LANGUAGE
from src.common.chat_history import ChatHistory
from .base import TransformersBaseLLM


class TransformersManualQwen3OmniLLM(TransformersBaseLLM):
    """
    vision/speech to text+speech voice chat
    """

    TAG = "llm_transformers_manual_qwen3omni"

    # Voice settings
    RATE = 24000
    SPEAKER_LIST = ["Chelsie", "Ethan"]

    def __init__(self, **args) -> None:
        self.args = Qwen3TransformersVisionVoiceLMArgs()
        self.args.update(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        self.thinker_args = TransformersLMArgs(**self.args.thinker_args)
        self.talker_args = TransformersLMArgs(**self.args.talker_args)
        self.code2wav_args = Qwen3OmniCode2WavArgs(**self.args.code2wav_args)
        logging.info(f"Model args: {args}")
        logging.info(f"Model thinker_args: {self.thinker_args}")
        logging.info(f"Model talker_args: {self.talker_args}")
        logging.info(f"Model code2wav_args: {self.code2wav_args}")
        config = AutoConfig.from_pretrained(self.args.lm_model_name_or_path)
        config.enable_audio_output = True
        if self.args.disable_talker is True:
            config.enable_audio_output = False

        if self.args.lm_device_map:
            self._model: Qwen3OmniMoeForConditionalGenerationStreaming = (
                Qwen3OmniMoeForConditionalGenerationStreaming.from_pretrained(
                    self.args.lm_model_name_or_path,
                    dtype=self.args.lm_torch_dtype,
                    #!NOTE: https://github.com/huggingface/transformers/issues/20896
                    # device_map for multi cpu/gpu with accelerate
                    device_map=self.args.lm_device_map,
                    attn_implementation=self.args.lm_attn_impl,
                    trust_remote_code=True,
                    config=config,
                ).eval()
            )
        else:
            self._model: Qwen3OmniMoeForConditionalGenerationStreaming = (
                Qwen3OmniMoeForConditionalGenerationStreaming.from_pretrained(
                    self.args.lm_model_name_or_path,
                    dtype=self.args.lm_torch_dtype,
                    attn_implementation=self.args.lm_attn_impl,
                    trust_remote_code=True,
                    config=config,
                )
                .eval()
                .to(self.args.lm_device)
            )

        self._tokenizer = Qwen3OmniMoeProcessor.from_pretrained(
            self.args.lm_model_name_or_path,
            use_fast=True,
        )

        self.session_chat_history: Dict[str, ChatHistory] = ThreadSafeDict()

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        logging.info(
            f"Warming up {self.__class__.__name__} device: {self._model.device} with {self.args.warmup_steps} steps"
        )

        dummy_msgs = [
            {
                "role": self.args.init_chat_role,
                "content": [
                    {
                        "type": "text",
                        "text": self.args.init_chat_prompt or self.CHAT_SYS_PROMPT,
                    }
                ],
            },
            {
                "role": self.args.user_role,
                "content": [
                    {"type": "text", "text": self.args.warmup_prompt or "请简单介绍下自己"}
                ],
            },
        ]
        # Preparation for inference
        text = self._tokenizer.apply_chat_template(
            dummy_msgs, tokenize=False, add_generation_prompt=True
        )
        if self.args.verbose:
            print(f"Warmup prompt token: {text}")
        audios, images, videos = process_mm_info(dummy_msgs, use_audio_in_video=False)
        inputs = self._tokenizer(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        if "cuda" in str(self._model.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for step in range(self.args.warmup_steps):
            streamer = self._model.generate_stream(
                inputs,
                use_audio_in_video=False,
                thinker_max_tokens_per_step=self.thinker_args.lm_gen_max_tokens_per_step,
                thinker_max_new_tokens=15,
                thinker_top_k=self.thinker_args.lm_gen_top_k,
                thinker_top_p=self.thinker_args.lm_gen_top_p,
                thinker_temperature=self.thinker_args.lm_gen_temperature,
                thinker_repetition_penalty=self.thinker_args.lm_gen_repetition_penalty,
                thinker_eos_token_ids=self.args.thinker_eos_token_ids,
                thinker_stop_strings_per_step=self.args.thinker_stop_strings_per_step,
                tokenizer=self._tokenizer.tokenizer,
                return_audio=self._model.has_talker,
                speaker=self.args.speaker,
                talker_top_k=self.talker_args.lm_gen_top_k,
                talker_top_p=self.talker_args.lm_gen_top_p,
                talker_temperature=self.talker_args.lm_gen_temperature,
                talker_repetition_penalty=self.talker_args.lm_gen_repetition_penalty,
                talker_min_new_tokens=self.talker_args.lm_gen_min_new_tokens,
                talker_max_new_tokens=self.talker_args.lm_gen_max_new_tokens,
                token2wav_kwargs=self.code2wav_args.__dict__,
                skip_token_ids=self.skip_token_ids(),
            )
            times = []
            start_time = time.perf_counter()
            text = ""
            for i, chunk in enumerate(streamer):
                times.append(time.perf_counter() - start_time)
                if "thinker_ids" in chunk:
                    text = self._tokenizer.decode(chunk["thinker_ids"][0], skip_special_tokens=True)
                    logging.info(f"{i} chunk: {text} , warmup time: {times[i]} s")
                if "talker_wav" in chunk:
                    logging.info(
                        f"{i} chunk: {text} | {chunk['talker_wav'].shape} , warmup time: {times[i]} s"
                    )
                start_time = time.perf_counter()
            if len(times) > 0:
                logging.info(
                    f"step {step} warmup TTFT(chunk) time: {times[0]} s | total: {sum(times)} s"
                )
            else:
                logging.warning(f"step {step} warmup no generate stream")
            step += 1

        if "cuda" in str(self._model.device):
            end_event.record()
            torch.cuda.synchronize()
            logging.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    @lru_cache
    def skip_token_ids(self):
        token_ids = []
        for i in ",;.?，；。？！":
            # for i in ",.":
            token_id = self._tokenizer.tokenizer.encode(i)
            token_ids.extend(token_id)
        return token_ids

    def get_prompt(self, session: Session) -> list:
        prompt = []
        if isinstance(session.ctx.state["prompt"], list):
            prompt = session.ctx.state["prompt"]
        return prompt

    @torch.no_grad()
    def generate(self, session: Session, **kwargs):
        """
        - prompt:
        [
            {"type": "image", "image": url / path / base64 / nparray},
            {"type": "video", "video": url / path / base64 / nparray},
            {"type": "audio", "audio": url / path / base64 / nparray},
            {"type": "text", "text": str},
        ]

        - return Generator[dict, None, None]:
        {
            "text": str,
            "audio_wav": torch.Tensor,# (T,)
        }
        """
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = self.get_prompt(session)

        message = {"role": self.args.user_role, "content": prompt}
        self.add_chat_history(session, message)
        messages = self.get_session_chat_history(session.ctx.client_id)
        logging.debug(f"messages: {messages}")

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=kwargs.get("use_audio_in_video", False)
        )
        if self.args.verbose:
            print(f"prompt token: {text}")
        {
            logging.debug(f"audios[{i}]: {item.shape}") for i, item in enumerate(audios)
        } if audios else logging.debug(audios)
        {
            logging.debug(f"images[{i}]: {item}") for i, item in enumerate(images)
        } if images else logging.debug(images)
        {
            logging.debug(f"videos[{i}]: {item.shape}") for i, item in enumerate(videos)
        } if videos else logging.debug(videos)

        inputs = self._tokenizer(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=kwargs.get("use_audio_in_video", False),
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)
        for k, v in inputs.items():
            logging.debug(f"{k}: {v.shape}")

        return_audio = kwargs.get("return_audio", self._model.has_talker)
        gen_assistant_text = ""
        try:
            if not return_audio:  # text / vision(image/video) / audio / text + image -> text
                for item in self.thinker_stream(
                    inputs,
                    use_audio_in_video=kwargs.get("use_audio_in_video", False),
                    thinker_top_k=kwargs.get("thinker_top_k", None)
                    or self.thinker_args.lm_gen_top_k,
                    thinker_top_p=kwargs.get("thinker_top_p", None)
                    or self.thinker_args.lm_gen_top_p,
                    thinker_temperature=kwargs.get("thinker_temperature", None)
                    or self.thinker_args.lm_gen_temperature,
                    thinker_repetition_penalty=kwargs.get("thinker_repetition_penalty", None)
                    or self.thinker_args.lm_gen_repetition_penalty,
                    thinker_min_new_tokens=kwargs.get("thinker_min_new_tokens", None)
                    or self.thinker_args.lm_gen_min_new_tokens,
                    thinker_max_new_tokens=kwargs.get("thinker_max_new_tokens", None)
                    or self.thinker_args.lm_gen_max_new_tokens,
                ):
                    gen_assistant_text += item["text"]
                    yield item
            else:  # text / vision(image/video) / audio / text + image -> text + audio
                stream = self._model.generate_stream(
                    inputs,
                    use_audio_in_video=kwargs.get("use_audio_in_video", False),
                    thinker_max_tokens_per_step=kwargs.get("thinker_max_tokens_per_step", None)
                    or self.thinker_args.lm_gen_max_tokens_per_step,
                    thinker_max_new_tokens=kwargs.get("thinker_max_new_tokens", None)
                    or self.thinker_args.lm_gen_max_new_tokens,
                    thinker_top_k=kwargs.get("thinker_top_k", None)
                    or self.thinker_args.lm_gen_top_k,
                    thinker_top_p=kwargs.get("thinker_top_p", None)
                    or self.thinker_args.lm_gen_top_p,
                    thinker_temperature=kwargs.get("thinker_temperature", None)
                    or self.thinker_args.lm_gen_temperature,
                    thinker_repetition_penalty=kwargs.get("thinker_repetition_penalty", None)
                    or self.thinker_args.lm_gen_repetition_penalty,
                    thinker_eos_token_ids=kwargs.get("thinker_eos_token_ids", None)
                    or self.args.thinker_eos_token_ids,
                    thinker_stop_strings_per_step=kwargs.get("thinker_stop_strings_per_step", None)
                    or self.args.thinker_stop_strings_per_step,
                    tokenizer=self._tokenizer.tokenizer,
                    return_audio=kwargs.get("return_audio", self._model.has_talker),
                    speaker=kwargs.get("speaker", None) or self.args.speaker,
                    talker_top_k=kwargs.get("talker_top_k", None) or self.talker_args.lm_gen_top_k,
                    talker_top_p=kwargs.get("talker_top_p", None) or self.talker_args.lm_gen_top_p,
                    talker_temperature=kwargs.get("talker_temperature", None)
                    or self.talker_args.lm_gen_temperature,
                    talker_repetition_penalty=kwargs.get("talker_repetition_penalty", None)
                    or self.talker_args.lm_gen_repetition_penalty,
                    talker_min_new_tokens=kwargs.get("talker_min_new_tokens", None)
                    or self.talker_args.lm_gen_min_new_tokens,
                    talker_max_new_tokens=kwargs.get("talker_max_new_tokens", None)
                    or self.talker_args.lm_gen_max_new_tokens,
                    token2wav_kwargs=kwargs.get("token2wav_kwargs", None)
                    or self.code2wav_args.__dict__,
                    skip_token_ids=self.skip_token_ids(),
                )

                gen_text = ""
                for chunk in stream:
                    if "thinker_ids" in chunk:
                        text = self._tokenizer.decode(
                            chunk["thinker_ids"][0], skip_special_tokens=True
                        )
                        if gen_text != text:
                            gen_text = text
                            gen_assistant_text += text
                        yield {"text": text}

                    if "talker_wav" in chunk:
                        # audio_bytes = (
                        #    (chunk["talker_wav"].float().detach().cpu().numpy() * 32768)
                        #    .astype(np.int16)
                        #    .tobytes()
                        # )
                        yield {"audio_wav": chunk["talker_wav"]}
        except Exception as e:
            tb_str = traceback.format_exc()
            logging.error(f"Exception: {e}; traceback: {tb_str}")

        self.add_chat_history(
            session, {"role": "assistant", "content": [{"text": gen_assistant_text}]}
        )

    @torch.no_grad()
    def thinker_stream(
        self,
        inputs: dict,
        use_audio_in_video: bool = False,
        thinker_top_k: int = 40,
        thinker_top_p: float = 0.8,
        thinker_temperature: float = 0.9,
        thinker_repetition_penalty: float = 1.05,
        thinker_min_new_tokens: int = 1,
        thinker_max_new_tokens: int = 1024,
    ):
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            thinker_do_sample=True if thinker_temperature > 0.0 else False,
            temperature=thinker_temperature,
            top_k=thinker_top_k,
            top_p=thinker_top_p,
            repetition_penalty=thinker_repetition_penalty,
            min_new_tokens=thinker_min_new_tokens,
            max_new_tokens=thinker_max_new_tokens,
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        times = []
        start_time = perf_counter()
        for new_text in streamer:
            times.append(perf_counter() - start_time)
            generated_text += new_text
            if new_text == "":
                continue
            yield {"text": new_text}
            start_time = perf_counter()

        logging.info(
            f"thinker generate [{generated_text}] TTFT: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s"
        )


class TransformersManualVisionVoiceQwen3OmniLLM(TransformersManualQwen3OmniLLM):
    """
    vision + speech to speech voice chat

    - vision + speech -> text + speech
    """

    TAG = "llm_transformers_manual_qwen3omni_vision_voice"
    USER_SYSTEM_PROMPT = "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen."
    CHAT_SYS_PROMPT = f"{USER_SYSTEM_PROMPT} You are a virtual voice assistant with no gender or age.\nYou are communicating with the user.\nIn user messages, “I/me/my/we/our” refer to the user and “you/your” refer to the assistant. In your replies, address the user as “you/your” and yourself as “I/me/my”; never mirror the user’s pronouns—always shift perspective. Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\nInteract with users using short(no more than 50 words), brief, straightforward language, maintaining a natural tone.\nNever use formal phrasing, mechanical expressions, bullet points, overly structured language. \nYour output must consist only of the spoken content you want the user to hear. \nDo not include any descriptions of actions, emotions, sounds, or voice changes. \nDo not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. \nYou must answer users' audio or text questions, do not directly describe the video content. \nYou should communicate in the same language strictly as the user unless they request otherwise.\nWhen you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.\nKeep replies concise and conversational, as if talking face-to-face."

    def __init__(self, **args) -> None:
        args["disable_talker"] = False
        language = "English"
        if args.get("lm_language_code", "") in TO_LLM_LANGUAGE:
            language = TO_LLM_LANGUAGE[args.get("lm_language_code", "")]
        self.CHAT_SYS_PROMPT += f"\nPlease reply to my message in {language}."
        super().__init__(**args)
