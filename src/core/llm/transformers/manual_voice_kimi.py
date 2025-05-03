import logging
import os
import sys
from threading import Thread
import time
from typing import List

import torch


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../KimiAudio"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/KimiAudio"))
    from deps.KimiAudio.kimia_infer.models.detokenizer import PrefixStreamingFlowMatchingDetokenizer
    from deps.KimiAudio.kimia_infer.utils.special_tokens import instantiate_extra_tokens
    from deps.KimiAudio.kimia_infer.utils.sampler import KimiASampler
    from deps.KimiAudio.kimia_infer.api.prompt_manager import KimiAPromptManager
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Kimi-Audio, you need to `pip install achatbot[llm_transformers_manual_voice_kimi]`. "
    )
    raise Exception(f"Missing module: {e}")

from src.common.types import ASSETS_DIR
from src.common.utils.helper import get_device
from src.common.session import Session
from src.types.omni.kimi_voice import KimiAudioTransformersVoiceLMArgs, KimiAudioDeTokenizerArgs
from .base import TransformersBaseLLM


class TransformersManualVoiceKimiLLM(TransformersBaseLLM):
    """
    https://github.com/ai-bot-pro/achatbot/pull/144
    """

    TAG = "llm_transformers_manual_kimi_voice"
    RATE = 24000

    def __init__(self, **args):
        self.args = KimiAudioTransformersVoiceLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info(f"args: {self.args}")

        # 1. load llm with double heads (text|audio) for sampling
        if self.args.lm_device_map:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                attn_implementation=self.args.lm_attn_impl,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                trust_remote_code=True,
            ).eval()
        else:
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=self.args.lm_torch_dtype,
                    attn_implementation=self.args.lm_attn_impl,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.args.lm_device)
            )

        model_config = self._model.config
        logging.debug(f"model config: {model_config}")
        self.kimia_token_offset = model_config.kimia_token_offset

        # 2. load Audio-Tokenizer: align text/audio tokenizer with delay tokens
        self.prompt_manager = KimiAPromptManager(
            model_path=self.args.lm_model_name_or_path,
            kimia_token_offset=self.kimia_token_offset,
        )
        self.extra_tokens = instantiate_extra_tokens(self.prompt_manager.text_tokenizer)
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]
        self.text_audio_delay_tokens_cn = 6

        # 3. load Audio-DeTokenizer code2wav: dit flow match + bigvgan vocoder model
        self.detokenizer = None
        if self.args.is_load_detokenizer is True:
            logging.info("Loading detokenizer")
            # need to compile extension moudules for the first time, it may take several minutes.
            fm_model_config = os.path.join(
                self.args.lm_model_name_or_path, "audio_detokenizer", "config.yaml"
            )
            fm_ckpt_path = os.path.join(
                self.args.lm_model_name_or_path, "audio_detokenizer", "model.pt"
            )
            bigvgan_config_file = os.path.join(
                self.args.lm_model_name_or_path, "vocoder", "config.json"
            )
            bigvgan_ckpt_path = os.path.join(self.args.lm_model_name_or_path, "vocoder", "model.pt")

            self.code2wav_args = KimiAudioDeTokenizerArgs(**self.args.code2wav_args)
            self.code2wav_args.device = self.code2wav_args.device or get_device()
            self.detokenizer = PrefixStreamingFlowMatchingDetokenizer.from_pretrained(
                vocoder_config=bigvgan_config_file,
                vocoder_ckpt=bigvgan_ckpt_path,
                fm_config=fm_model_config,
                fm_ckpt=fm_ckpt_path,
                device=self.code2wav_args.device,
                look_ahead_tokens=self.code2wav_args.look_ahead_tokens,
                max_prompt_chunk=self.code2wav_args.max_prompt_chunk,
                max_kv_cache_tokens=self.code2wav_args.max_kv_cache_tokens,
                use_cfg=self.code2wav_args.use_cfg,
                use_cfg_rescale=self.code2wav_args.use_cfg_rescale,
                cfg_init=self.code2wav_args.cfg_init,
                cfg_scale=self.code2wav_args.cfg_scale,
                cfg_schedule=self.code2wav_args.cfg_schedule,
            )

        self.warmup()

    @torch.inference_mode()
    def warmup(self):
        if self.args.warmup_steps < 1:
            return
        logging.info(f"Warming up {self.__class__.__name__} device: {self._model.device}")
        dummy_input_text = self.args.warnup_prompt.strip()
        text_token_ids = [self.extra_tokens.kimia_text_blank]  # user
        token_ids = self.prompt_manager.text_tokenizer.encode(
            dummy_input_text, bos=False, eos=False
        )
        text_token_ids.extend(token_ids)
        text_token_ids.append(self.extra_tokens.kimia_text_blank)  # end
        text_token_ids.append(self.extra_tokens.kimia_text_blank)  # assistant
        audio_token_ids = (
            [self.extra_tokens.user_msg_start]  # user
            + [self.extra_tokens.kimia_text_blank] * len(token_ids)
            + [self.extra_tokens.msg_end]  # end
            + [self.extra_tokens.assistant_msg_start]  # assistant
        )
        is_continuous_mask = [False] * len(audio_token_ids)
        # max_new_tokens = int(12.5 * 120) - len(audio_token_ids)
        max_new_tokens = 50

        if "cuda" in str(self._model.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for step in range(self.args.warmup_steps):
            streamer = self.gen_token_stream(
                audio_input_ids=torch.tensor(
                    [audio_token_ids], dtype=torch.long, device=self._model.device
                ),
                text_input_ids=torch.tensor(
                    [text_token_ids], dtype=torch.long, device=self._model.device
                ),
                is_continuous_mask=torch.tensor(
                    [is_continuous_mask], dtype=torch.bool, device=self._model.device
                ),
                audio_continous_features=[],
                output_type="both",
                max_new_tokens=max_new_tokens,
            )

            text = ""
            times = []
            start_time = time.perf_counter()
            for text_token_id, _ in streamer:
                times.append(time.perf_counter() - start_time)
                text += self.prompt_manager.text_tokenizer.decode([text_token_id.item()])
                start_time = time.perf_counter()

            if len(times) > 0:
                logging.info(
                    f"step {step} | gen_text: {text} | warmup TTFT time: {times[0]} s | total: {sum(times)} s"
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

    def gen_token_stream(
        self,
        audio_input_ids: torch.Tensor,  # input audio tokens
        text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
        is_continuous_mask: torch.Tensor = None,
        audio_continous_features: List[torch.Tensor] = [],
        sampler: KimiASampler = KimiASampler(),
        output_type: str = "text",
        max_new_tokens: int = 128,
    ):
        assert output_type in ["text", "both"], f"output_type: {output_type}"

        is_output_audio = output_type == "both" and self.args.is_load_detokenizer is True

        text_stream_is_finished = False
        previous_audio_tokens = torch.zeros(
            (max_new_tokens,),
            dtype=torch.int,
            device=self._model.device,
        )
        text_previous_tokens = torch.zeros(
            (max_new_tokens,),
            dtype=torch.int,
            device=self._model.device,
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = (
            torch.arange(0, decoder_input_audio_ids.shape[1], device=self._model.device)
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = audio_continous_features or None
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1

        # one bye one generate, until eos or max_new_tokens
        for i in range(max_new_tokens):
            # https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/main/modeling_moonshot_kimia.py#L850
            # https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/main/config.json
            # use_cache=True
            audio_logits, text_logits, past_key_values = self._model.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )

            # Sample text token using the sampler
            next_text_token_id = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )
            # Sample audio token using the sampler
            next_audio_token_id = (
                sampler.sample_audio_logits(
                    audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
                )
                if i >= self.text_audio_delay_tokens_cn and is_output_audio
                else torch.Tensor([self.extra_tokens.kimia_text_blank])
                .long()
                .to(self._model.device)
            )

            if text_stream_is_finished:
                next_text_token_id.fill_(self.extra_tokens.kimia_text_blank)
            elif next_text_token_id.item() == self.extra_tokens.kimia_text_eos:
                text_stream_is_finished = True
            audio_stream_is_finished = next_audio_token_id.item() in self.eod_ids

            yield (next_text_token_id, next_audio_token_id)  # (1,) (1,)

            if is_output_audio is False and text_stream_is_finished:
                break
            if is_output_audio is True and text_stream_is_finished and audio_stream_is_finished:
                break

            text_previous_tokens[i : i + 1] = next_text_token_id
            previous_audio_tokens[i : i + 1] = next_audio_token_id

            decoder_input_audio_ids = next_audio_token_id.unsqueeze(1)  # (1,1)
            decoder_input_text_ids = next_text_token_id.unsqueeze(1)  # (1,1)
            last_position_id += 1
            decoder_position_ids = (
                torch.zeros(1, 1, device=self._model.device)
                .fill_(last_position_id)
                .long()
                .view(1, 1)
            )  # (1,1)
            decoder_input_whisper_feature = None
            decoder_is_continuous_mask = None

    # @torch.no_grad()
    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        prompt: List[Dict]
        prompt[i] = {
            "role": "user" | "assistant",
            "message_type": "text" | "audio",
            "content": str | torch.Tensor
        }

        - return Generator[dict, None, None]:
        {
            "text": str,
            "audio_wav": torch.Tensor,# (T,)
        }
        """
        output_type = session.ctx.state.get("output_type", "both")
        assert output_type in ["text", "both"], f"output_type: {output_type}"

        # process prompt
        messages = session.ctx.state.get("prompt", [])
        prompt = self.prompt_manager.get_prompt(messages, output_type=output_type)
        audio_input_ids, text_input_ids, is_continuous_mask = prompt.to_tensor()
        audio_continuous_features = prompt.continuous_feature
        audio_input_ids = audio_input_ids.to(self._model.device)
        text_input_ids = text_input_ids.to(self._model.device)
        is_continuous_mask = is_continuous_mask.to(self._model.device)
        audio_continuous_features = [f.to(self._model.device) for f in audio_continuous_features]

        max_new_tokens = kwargs.get("max_new_tokens", -1)
        if output_type == "both":
            # magic number : 12.5 * 120 ?
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        elif output_type == "text" and max_new_tokens < 0:
            max_new_tokens = 7500 - audio_input_ids.shape[1]

        chunk_size = kwargs.get("chunk_size", 30)

        # 2 heads for text and audio to do sampling
        sampler = KimiASampler(
            audio_top_k=kwargs.get("audio_top_k", 10),
            audio_temperature=kwargs.get("audio_temperature", 0.8),
            audio_repetition_penalty=kwargs.get("audio_repetition_penalty", 1.0),
            audio_repetition_window_size=kwargs.get("audio_repetition_window_size", 64),
            text_top_k=kwargs.get("text_top_k", 5),
            text_temperature=kwargs.get("text_temperature", 0.0),
            text_repetition_penalty=kwargs.get("text_repetition_penalty", 1.0),
            text_repetition_window_size=kwargs.get("text_repetition_window_size", 16),
        )
        streamer = self.gen_token_stream(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            is_continuous_mask=is_continuous_mask,
            audio_continous_features=audio_continuous_features,
            sampler=sampler,
            output_type=output_type,
            max_new_tokens=max_new_tokens,
        )

        audio_text = ""
        audio_vq_codes = []
        times = []
        audio_chunk_times = []
        start_time = time.perf_counter()

        for text_token_id, audio_token_id in streamer:
            times.append(time.perf_counter() - start_time)

            text_token_id = text_token_id.item()
            audio_token_id = audio_token_id.item()

            if output_type == "text" and text_token_id == self.extra_tokens.kimia_text_eos:
                break

            if (
                text_token_id != self.extra_tokens.kimia_text_eos
                and text_token_id < self.kimia_token_offset
                and text_token_id != self.extra_tokens.kimia_text_blank
            ):
                # https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/main/tokenization_kimia.py
                # no skip_special_tokens args :)
                text = self.prompt_manager.text_tokenizer.decode([text_token_id])
                yield {"text": text}
                audio_text += text

            if output_type == "text":
                continue

            if audio_token_id < self.kimia_token_offset:
                continue
            if audio_token_id in self.eod_ids:
                break

            audio_vq_code = audio_token_id - self.kimia_token_offset
            audio_vq_codes.append(audio_vq_code)
            if len(audio_vq_codes) % chunk_size == 0:
                logging.info(f"chunk audio_vq_codes: {audio_vq_codes}")
                start_time = time.perf_counter()
                gen_speech = self.detokenizer.detokenize_streaming(
                    torch.tensor(audio_vq_codes).unsqueeze(0).long().to(self.code2wav_args.device),
                    is_final=False,
                    upsample_factor=4,
                )
                audio_chunk_times.append(time.perf_counter() - start_time)
                audio_vq_codes = []
                yield {"audio_wav": gen_speech}

            start_time = time.perf_counter()

        if len(audio_vq_codes) > 0:
            logging.info(f"last vq_codes: {audio_vq_codes}")
            start_time = time.perf_counter()
            gen_speech = self.detokenizer.detokenize_streaming(
                torch.tensor(audio_vq_codes).unsqueeze(0).long().to(self.code2wav_args.device),
                is_final=True,
                upsample_factor=4,
            )
            audio_chunk_times.append(time.perf_counter() - start_time)
            yield {"audio_wav": gen_speech}

        logging.info(
            f"text [{audio_text}] TTFT: {times[0]} s | total: {sum(times)} s | len: {len(times)} | avg: {sum(times)/len(times)} s"
        )
        if len(audio_chunk_times) > 0:
            logging.info(
                f"audio TTFT(chunk): {audio_chunk_times[0]} s | total: {sum(audio_chunk_times)} s | len: {len(audio_chunk_times)} | avg: {sum(audio_chunk_times)/len(audio_chunk_times)} s"
            )


class TransformersManualAudioKimiLLM(TransformersManualVoiceKimiLLM):
    """
    audio understanding

    - speech -> text
    """

    TAG = [
        "llm_transformers_manual_kimi_audio_asr",
    ]

    def __init__(self, **args) -> None:
        args["is_load_detokenizer"] = False
        super().__init__(**args)

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        for item in super().generate(session, **kwargs):
            text = item.pop("text", "")
            if text == "":
                continue
            yield text


class TransformersManualTextVoiceKimiLLM(TransformersManualVoiceKimiLLM):
    """
    text to speech voice chat

    - text -> text + speech
    """

    TAG = "llm_transformers_manual_kimi_text_voice"

    def __init__(self, **args) -> None:
        super().__init__(**args)
