import logging
from threading import Thread
from time import perf_counter
import time
from typing import Generator, Optional

import numpy as np
import torch


try:
    from qwen_omni_utils import process_mm_info
    from transformers import (
        AutoConfig,
        AutoProcessor,
        TextIteratorStreamer,
    )
    from src.thirdparty.qwen2_code2wav.engine import Code2WavEngine
    from src.thirdparty.qwen2_code2wav import Code2WavEngineConfig, Code2WavGenerationConfig
    from src.core.llm.transformers.models.qwen2_5_omni import (
        Qwen2_5OmniForConditionalGenerationStreaming,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Qwen2.5Omni, you need to `pip install achatbot[llm_transformers_manual_vision_voice_qwen]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.utils.helper import ThreadSafeDict
from src.core.llm.transformers.streamer import TokenStreamer
from src.common.random import set_all_random_seed
from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.types.omni.qwen2_vision_voice import Qwen2_5TransformersVisionVoiceLMArgs
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


class TransformersManualQwen2_5OmniLLM(TransformersBaseLLM):
    TAG = "llm_transformers_manual_qwen2_5omni"

    # NOTE: if want to generate speech, need use this system prompt to generate speech
    SPEECH_SYS_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    # Voice settings
    SPEAKER_LIST = ["Chelsie", "Ethan"]
    DEFAULT_SPEAKER = "Chelsie"
    RATE = 24000

    def __init__(self, **args) -> None:
        self.args = Qwen2_5TransformersVisionVoiceLMArgs(**args)
        self.thinker_args = TransformersLMArgs(**self.args.thinker_args)
        self.talker_args = TransformersLMArgs(**self.args.talker_args)
        self.code2wav_args = Code2WavEngineConfig(**self.args.code2wav_args)
        logging.info(f"Model args: {args}")
        logging.info(f"Model thinker_args: {self.thinker_args}")
        logging.info(f"Model talker_args: {self.talker_args}")
        logging.info(f"Model code2wav_args: {self.code2wav_args}")
        config = AutoConfig.from_pretrained(self.args.lm_model_name_or_path)
        config.enable_audio_output = True
        if self.args.disable_talker is True:
            config.enable_audio_output = False

        if self.args.lm_device_map:
            self._model: Qwen2_5OmniForConditionalGenerationStreaming = (
                Qwen2_5OmniForConditionalGenerationStreaming.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=self.args.lm_torch_dtype,
                    #!NOTE: https://github.com/huggingface/transformers/issues/20896
                    # device_map for multi cpu/gpu with accelerate
                    device_map=self.args.lm_device_map,
                    attn_implementation=self.args.lm_attn_impl,
                    trust_remote_code=True,
                    config=config,
                ).eval()
            )
        else:
            self._model: Qwen2_5OmniForConditionalGenerationStreaming = (
                Qwen2_5OmniForConditionalGenerationStreaming.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=self.args.lm_torch_dtype,
                    attn_implementation=self.args.lm_attn_impl,
                    trust_remote_code=True,
                    config=config,
                )
                .eval()
                .to(self.args.lm_device)
            )

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a
        # token count range of 256-1280, to balance speed and memory usage.
        self._tokenizer = AutoProcessor.from_pretrained(
            self.args.lm_model_name_or_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            trust_remote_code=True,
        )

        # use sliding window code2wav
        self.code2wav_engine: Code2WavEngine = None
        if self.args.is_use_sliding_window_code2wav is True:
            self.code2wav_engine = Code2WavEngine(**self.args.code2wav_args)

        self.chat_history_dict = ThreadSafeDict()

        self.warmup()

    def chat_history(self, session: Session, **kwargs) -> ChatHistory:
        session_id = session.ctx.client_id
        if not self.chat_history_dict.get(session_id):
            chat_history = ChatHistory(
                kwargs.get("chat_history_size", None) or self.args.chat_history_size
            )
            init_chat_role = kwargs.get("init_chat_role", None) or self.args.init_chat_role
            init_chat_prompt = (
                kwargs.get("init_chat_prompt", self.args.init_chat_prompt) or self.SPEECH_SYS_PROMPT
            )
            if init_chat_role:
                sys_msg = {
                    "role": init_chat_role,
                    "content": [
                        {
                            "type": "text",
                            "text": init_chat_prompt,
                        }
                    ],
                }
                chat_history.init(sys_msg)
            self.chat_history_dict.set(session_id, chat_history)

        return self.chat_history_dict.get(session_id)

    def warmup(self):
        if self.args.warmup_steps < 0:
            return
        logging.info(f"Warming up {self.__class__.__name__} device: {self._model.device}")

        dummy_msgs = [
            {
                "role": self.args.init_chat_role,
                "content": [
                    {
                        "type": "text",
                        "text": self.args.init_chat_prompt or self.SPEECH_SYS_PROMPT,
                    }
                ],
            },
            {
                "role": self.args.user_role,
                "content": [
                    {"type": "text", "text": self.args.warnup_prompt or "what's your name?"}
                ],
            },
        ]
        # Preparation for inference
        text = self._tokenizer.apply_chat_template(
            dummy_msgs, tokenize=False, add_generation_prompt=True
        )
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
                **inputs,
                use_audio_in_video=False,
                thinker_max_tokens_per_step=self.thinker_args.lm_gen_max_tokens_per_step,
                thinker_max_new_tokens=self.thinker_args.lm_gen_max_new_tokens,
                thinker_top_k=self.thinker_args.lm_gen_top_k,
                thinker_top_p=self.thinker_args.lm_gen_top_p,
                thinker_temperature=self.thinker_args.lm_gen_temperature,
                thinker_repetition_penalty=self.thinker_args.lm_gen_repetition_penalty,
                thinker_eos_token_ids=self.args.thinker_eos_token_ids,
                return_audio=self._model.has_talker,
                speaker=self.args.speaker,
                talker_top_k=self.talker_args.lm_gen_top_k,
                talker_top_p=self.talker_args.lm_gen_top_p,
                talker_temperature=self.talker_args.lm_gen_temperature,
                talker_repetition_penalty=self.talker_args.lm_gen_repetition_penalty,
                talker_min_new_tokens=self.talker_args.lm_gen_min_new_tokens,
                talker_max_new_tokens=self.talker_args.lm_gen_max_new_tokens,
                talker_eos_token_ids=self.args.talker_eos_token_ids,
                code2wav_num_steps=self.code2wav_args.num_steps,
                code2wav_guidance_scale=self.code2wav_args.guidance_scale,
                code2wav_sway_coefficient=self.code2wav_args.sway_coefficient,
                code2wav_chunk_stream_func=self.code2wav_sliding_window_chunk_stream
                if self.args.is_use_sliding_window_code2wav
                else None,
            )
            times = []
            start_time = time.perf_counter()
            for _ in streamer:
                times.append(time.perf_counter() - start_time)
                start_time = time.perf_counter()
            logging.info(f"step {step} warnup TTFT(chunk) time: {times[0]} s")
            step += 1

        if "cuda" in str(self._model.device):
            end_event.record()
            torch.cuda.synchronize()
            logging.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def code2wav_sliding_window_chunk_stream(
        self,
        talker_streamer: TokenStreamer,
        speaker: str = DEFAULT_SPEAKER,
        talker_eos_token_ids: list[int] = [8292, 8294],
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]:
        """
        code2wav sliding window streaming
        """
        talker_eos_token_ids = talker_eos_token_ids or self.args.talker_eos_token_ids
        prev_generated = None
        progress = 0
        finished = False
        code2wav_times = []
        talker_generate_codes = []
        times = []
        start_time = perf_counter()
        for token_id in talker_streamer:
            times.append(perf_counter() - start_time)
            start_time = perf_counter()
            if token_id in talker_eos_token_ids:
                finished = True
            talker_generate_codes.append(token_id)
            prev_generated, wav = self.code2wav_engine.step_generate_waveform(
                talker_generate_codes,
                voice_type=speaker,
                prev_generated=prev_generated,
                progress=progress,
                finished=finished,
                gen_args=Code2WavGenerationConfig(
                    num_steps=kwargs.get("code2wav_num_steps") or self.code2wav_args.num_steps,
                    guidance_scale=kwargs.get("code2wav_guidance_scale")
                    or self.code2wav_args.guidance_scale,
                    sway_coefficient=kwargs.get("code2wav_sway_coefficient")
                    or self.code2wav_args.sway_coefficient,
                ),
            )
            if wav is not None:
                progress += 1
                code2wav_times.append(perf_counter() - start_time)
                yield wav.detach()  # (T,)

            start_time = perf_counter()

        logging.info(
            f"talker generate first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s"
        )
        logging.info(
            f"code2wav sliding window streaming first chunk time: {code2wav_times[0]} s | cost: {sum(code2wav_times)} s"
        )

        torch.cuda.empty_cache()

    def get_prompt(self, session: Session) -> list:
        prompt = []
        if isinstance(session.ctx.state["prompt"], list):
            prompt = session.ctx.state["prompt"]
        return prompt

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        prompt:
        [
            {"type": "text", "text": str},
            {"type": "image", "image": url / path / base64 / nparray},
            {"type": "video", "video": url / path / base64 / nparray},
            {"type": "audio", "audio": url / path / base64 / nparray},
        ]
        """
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = self.get_prompt(session)

        message = {"role": "user", "content": prompt}
        session_chat_history = self.chat_history(session, **kwargs)
        session_chat_history.append(message)
        messages = session_chat_history.to_list()
        logging.info(f"messages: {messages}")

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=kwargs.get("use_audio_in_video", False)
        )
        logging.info(f"{text}, {audios}, {images}, {videos}")
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

        return_audio = kwargs.get("return_audio", self._model.has_talker)
        if not return_audio:
            return self.thinker_inference_stream(inputs)

        gen_args = dict(
            use_audio_in_video=kwargs.get("use_audio_in_video", False),
            thinker_max_tokens_per_step=kwargs.get("thinker_max_tokens_per_step", None)
            or self.thinker_args.lm_gen_max_tokens_per_step,
            thinker_max_new_tokens=kwargs.get("thinker_max_new_tokens", None)
            or self.thinker_args.lm_gen_max_new_tokens,
            thinker_top_k=kwargs.get("thinker_top_k", None) or self.thinker_args.lm_gen_top_k,
            thinker_top_p=kwargs.get("thinker_top_p", None) or self.thinker_args.lm_gen_top_p,
            thinker_temperature=kwargs.get("thinker_temperature", None)
            or self.thinker_args.lm_gen_temperature,
            thinker_repetition_penalty=kwargs.get("thinker_eos_token_ids", None)
            or self.thinker_args.lm_gen_repetition_penalty,
            thinker_eos_token_ids=kwargs.get("thinker_eos_token_ids", None)
            or self.args.thinker_eos_token_ids,
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
            talker_eos_token_ids=kwargs.get("talker_eos_token_ids", None)
            or self.args.talker_eos_token_ids,
            talker_skip_thinker_token_ids=kwargs.get("talker_skip_thinker_token_ids", None) or [],
            code2wav_num_steps=kwargs.get("code2wav_num_steps", None)
            or self.code2wav_args.num_steps,
            code2wav_guidance_scale=kwargs.get("code2wav_guidance_scale", None)
            or self.code2wav_args.guidance_scale,
            code2wav_sway_coefficient=kwargs.get("code2wav_sway_coefficient", None)
            or self.code2wav_args.sway_coefficient,
        )
        stream = self._model.generate_stream(
            **inputs,
            **gen_args,
            code2wav_chunk_stream_func=self.code2wav_sliding_window_chunk_stream
            if self.args.is_use_sliding_window_code2wav
            else None,
        )

        gen_text = ""
        all_gen_text = ""
        for chunk in stream:
            text = self._tokenizer.decode(chunk["thinker_ids"][0], skip_special_tokens=True)
            if gen_text != text:
                gen_text = text
                all_gen_text += text

            if "talker_wav" not in chunk:
                yield {"text": text}
            else:
                # audio_bytes = (
                #    (chunk["talker_wav"].float().detach().cpu().numpy() * 32768)
                #    .astype(np.int16)
                #    .tobytes()
                # )

                yield {"text": text, "audio_wav": chunk["talker_wav"]}

        session_chat_history.append({"role": "assistant", "content": [{"text": all_gen_text}]})
        self.chat_history_dict.set(session.ctx.client_id, session_chat_history)

    def thinker_inference_stream(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
            input_ids,
            attention_mask=attention_mask,
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
            yield {"text": new_text}
            start_time = perf_counter()

        logging.info(
            f"thinker generate [{generated_text}] TTFT: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s"
        )


class TransformersManualAudioQwen2_5OmniLLM(TransformersManualQwen2_5OmniLLM):
    """
    audio understanding

    - speech -> text
    """

    TAG = [
        "llm_transformers_manual_qwen2_5omni_audio_asr",
        "llm_transformers_manual_qwen2_5omni_audio_translation",
        "llm_transformers_manual_qwen2_5omni_audio_classification",
    ]

    def __init__(self, **args) -> None:
        args["disable_talker"] = True
        args["init_chat_prompt"] = args.get(
            "init_chat_prompt", "You are a speech recognition model"
        )
        if self.SELECTED_TAG == "llm_transformers_manual_qwen2_5omni_audio_asr":
            args["init_chat_prompt"] = "You are a speech recognition model"
        if self.SELECTED_TAG == "llm_transformers_manual_qwen2_5omni_audio_translation":
            args["init_chat_prompt"] = "You are a speech translation model"
        if self.SELECTED_TAG == "llm_transformers_manual_qwen2_5omni_audio_classification":
            args["init_chat_prompt"] = "You are a voice classification model."

        super().__init__(**args)


class TransformersManualVisionQwen2_5OmniLLM(TransformersManualQwen2_5OmniLLM):
    """
    vision only, vision understanding

    - vision -> text
    """

    TAG = "llm_transformers_manual_qwen2_5omni_vision"

    def __init__(self, **args) -> None:
        args["disable_talker"] = True
        args["init_chat_prompt"] = args.get("init_chat_prompt", "You are a helpful assistant.")
        super().__init__(**args)


class TransformersManualVisionVoiceQwen2_5OmniLLM(TransformersManualQwen2_5OmniLLM):
    """
    vision + speech to speech voice chat

    - vision + speech -> text + speech
    """

    TAG = "llm_transformers_manual_qwen2_5omni_vision_voice"

    def __init__(self, **args) -> None:
        args["disable_talker"] = False
        super().__init__(**args)


class TransformersManualTextVoiceQwen2_5OmniLLM(TransformersManualQwen2_5OmniLLM):
    """
    speech to speech voice chat

    - text -> text + speech
    """

    TAG = "llm_transformers_manual_qwen2_5omni_text_voice"

    def __init__(self, **args) -> None:
        args["disable_talker"] = False
        super().__init__(**args)


class TransformersManualVoiceQwen2_5OmniLLM(TransformersManualQwen2_5OmniLLM):
    """
    speech to speech voice chat

    - speech -> text + speech
    """

    TAG = "llm_transformers_manual_qwen2_5omni_audio_voice"

    def __init__(self, **args) -> None:
        args["disable_talker"] = False
        super().__init__(**args)
