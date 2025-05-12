import logging
import os
import re
import sys
from threading import Thread
import time
import traceback
from typing import List

import torch


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig

    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../VITAAudio"))
        sys.path.insert(1, os.path.join(cur_dir, "../../../VITAAudio/third_party/GLM-4-Voice/"))
        sys.path.insert(
            1, os.path.join(cur_dir, "../../../VITAAudio/third_party/GLM-4-Voice/cosyvoice/")
        )
        sys.path.insert(
            1,
            os.path.join(
                cur_dir, "../../../VITAAudio/third_party/GLM-4-Voice/third_party/Matcha-TTS/"
            ),
        )
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/VITAAudio"))
        sys.path.insert(
            1, os.path.join(cur_dir, "../../../../deps/VITAAudio/third_party/GLM-4-Voice/")
        )
        sys.path.insert(
            1,
            os.path.join(cur_dir, "../../../../deps/VITAAudio/third_party/GLM-4-Voice/cosyvoice/"),
        )
        sys.path.insert(
            1,
            os.path.join(
                cur_dir,
                "../../../../deps/VITAAudio/third_party/GLM-4-Voice/third_party/Matcha-TTS/",
            ),
        )

    from deps.VITAAudio.vita_audio.tokenizer import get_audio_tokenizer
    from deps.VITAAudio.vita_audio.data.processor.audio_processor import add_audio_input_contiguous
    from deps.VITAAudio.evaluation.get_chat_template import qwen2_chat_template as chat_template

except ModuleNotFoundError as e:
    ex_trace = traceback.format_exc()
    logging.error(f"Exception: {ex_trace}")
    logging.error(
        "In order to use VITA-Audio, you need to `pip install achatbot[llm_transformers_manual_voice_vita]`. "
    )
    raise Exception(f"Missing module: {e}")

from src.common.types import ASSETS_DIR
from src.common.utils.helper import get_device
from src.common.session import Session
from src.types.omni.vita_voice import VitaAudioTransformersVoiceLMArgs
from .base import TransformersBaseLLM
from .streamer import TextAudioIteratorStreamer


def find_audio_segments_regex(text):
    """
    Find all substrings between <|begin_of_audio|> and <|end_of_audio|> using regex.

    Args:
        text (str): The input string to search through

    Returns:
        list: A list of all found audio segments (substrings between the delimiters)
    """
    pattern = re.compile(r"<\|begin_of_audio\|>(.*?)<\|end_of_audio\|>", re.DOTALL)
    segments = pattern.findall(text)
    return [segment.strip() for segment in segments]


def extract_token_ids_as_int(text):
    pattern = re.compile(r"<\|audio_(\d+)\|>")
    token_ids = pattern.findall(text)
    return [int(id) for id in token_ids]


class TransformersManualTextVITALLM(TransformersBaseLLM):
    """
    https://github.com/ai-bot-pro/achatbot/pull/146

    - text chat: text -> text
    """

    TAG = "llm_transformers_manual_vita_text"
    RATE = 22050

    def __init__(self, **args):
        self.args = VitaAudioTransformersVoiceLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info(f"args: {self.args}")

        # 1. load llm with double heads (text|audio) for sampling
        # https://huggingface.co/VITA-MLLM/VITA-Audio-Boost/blob/main/modeling_qwen2.py#L781
        # https://huggingface.co/VITA-MLLM/VITA-Audio-Balance/blob/main/modeling_qwen2.py#L781
        # https://huggingface.co/VITA-MLLM/VITA-Audio-Plus-Vanilla/blob/main/modeling_qwen2.py#L834
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

        logging.info(f"model config: {self._model.config}")
        logging.info(f"{self._model.device=}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path,
            trust_remote_code=True,
            chat_template=chat_template,
        )

        self._model.generation_config = GenerationConfig.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True
        )

        self._model.generation_config.max_new_tokens = 8192
        self._model.generation_config.chat_format = "chatml"
        self._model.generation_config.max_window_size = 8192
        self._model.generation_config.use_cache = True
        # model.generation_config.use_cache = False
        self._model.generation_config.do_sample = False
        self._model.generation_config.temperature = 1.0
        self._model.generation_config.top_k = 50
        self._model.generation_config.top_p = 1.0
        self._model.generation_config.num_beams = 1
        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id
        logging.info(f"{self._model.generation_config=}")

        # 2. load audio tokenizer
        self.audio_tokenizer = get_audio_tokenizer(
            model_type=self.args.audio_tokenizer_type,
            rank=self.args.audio_tokenizer_rank,
            flow_path=self.args.flow_path,
            model_name_or_path=self.args.audio_tokenizer_model_path,
            sense_voice_model_path=self.args.sense_voice_model_path,
        )
        self.audio_tokenizer.load_model()

        self.add_generation_prompt = True
        self.default_system_message = []
        self.luke_system_message = [
            {
                "role": "system",
                "content": "Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner.",
            },
        ]
        self.audio_0_id = self._tokenizer("<|audio_0|>").input_ids[0]

        self.warmup()

    @torch.inference_mode()
    def warmup(self):
        if self.args.warmup_steps < 1:
            return
        logging.info(f"Warming up {self.__class__.__name__} device: {self._model.device}")
        dummy_input_text = self.args.warnup_prompt.strip()

        if "cuda" in str(self._model.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for step in range(self.args.warmup_steps):
            text = ""
            times = []
            start_time = time.perf_counter()
            for token in self.run_infer_stream(
                message=dummy_input_text,
                mode=None,
                do_sample=True,
                mtp_inference_mode=[8192, 0],
            ):
                times.append(time.perf_counter() - start_time)
                text += token
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

    def run_infer_stream(
        self,
        audio_path: str = None,
        prompt_audio_path: str = None,
        message="",
        mode="luke",
        do_sample=False,
        mtp_inference_mode=None,
    ):
        if prompt_audio_path is not None:
            system_message = [
                {
                    "role": "system",
                    "content": f"Your Voice: <|audio|>\n",
                },
            ]

        elif mode == "luke":
            system_message = self.luke_system_message

        else:
            system_message = self.default_system_message

        if prompt_audio_path is not None and self.audio_tokenizer.apply_to_role(
            "user", is_discrete=True
        ):
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(prompt_audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            system_message[-1]["content"] = system_message[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        if audio_path is not None:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message + "\n<|audio|>",
                },
            ]
        else:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message,
                },
            ]

        if audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            # print("discrete codec")
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )

        if (
            audio_path is not None or prompt_audio_path is not None
        ) and self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
            # print("contiguous codec")
            # contiguous codec
            audio_paths = []
            if audio_path is not None:
                audio_paths.append(audio_path)
            if prompt_audio_path is not None:
                audio_paths.append(prompt_audio_path)
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, audio_paths, self._tokenizer, self.audio_tokenizer
            )
        else:
            audios = None
            audio_indices = None

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self._model.device)

        # print("input", self._tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)
        attention_mask = torch.ones((1, input_ids.shape[1]), dtype=torch.int64).to(
            self._model.device
        )

        self._model.generation_config.do_sample = do_sample

        if mtp_inference_mode is not None:
            # print(f"{mtp_inference_mode=}")
            ori_mtp_inference_mode = self._model.generation_config.mtp_inference_mode
            self._model.generation_config.mtp_inference_mode = mtp_inference_mode

        streamer = TextAudioIteratorStreamer(self._tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audios=audios,
            audio_indices=audio_indices,
            streamer=streamer,
        )
        # print(
        #    f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape} audio_indices: {audio_indices} audios: {audios}"
        # )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)

        thread.start()

        for new_text in streamer:
            yield new_text

        if mtp_inference_mode is not None:
            self._model.generation_config.mtp_inference_mode = ori_mtp_inference_mode

    # @torch.no_grad()
    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        prompt: str | torch.Tensor

        - return Generator[str, None, None]:
        """
        # process prompt
        prompt = session.ctx.state.get("prompt", "")

        text = ""
        times = []
        start_time = time.perf_counter()
        for token in self.run_infer_stream(
            message=prompt,
            mode=None,
            do_sample=True,
            mtp_inference_mode=[8192, 0],
        ):
            times.append(time.perf_counter() - start_time)
            yield token
            text += token
            start_time = time.perf_counter()

        logging.info(
            f"text [{text}] TTFT: {times[0]} s | total: {sum(times)} s | len: {len(times)} | avg: {sum(times)/len(times)} s"
        )


class TransformersManualAudioVITALLM(TransformersManualTextVITALLM):
    """
    audio understanding

    - speech -> text
    """

    TAG = [
        "llm_transformers_manual_vita_audio_asr",
    ]

    def __init__(self, **args) -> None:
        args["flow_path"] = None
        super().__init__(**args)

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        prompt: str | torch.Tensor

        - return Generator[str, None, None]:
        """
        # process prompt
        prompt = session.ctx.state.get("prompt", None)
        assert prompt is not None

        times = []
        start_time = time.perf_counter()
        generated_text = ""
        for new_text in self.run_infer_stream(
            audio_path=prompt,
            message="Convert the speech to text.",
            mode=None,
        ):
            times.append(time.perf_counter() - start_time)
            generated_text += new_text
            yield new_text
            start_time = time.perf_counter()
        logging.info(
            f"\ngenerate [{generated_text}] first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s\n"
        )


class TransformersManualTextSpeechVITALLM(TransformersManualTextVITALLM):
    """
    tts: coverate text to speech

    - text -> speech
    """

    TAG = "llm_transformers_manual_vita_tts"

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.chunk_size_list = self.args.chunk_size_list  # like tcp cwnd

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        prompt: str | torch.Tensor

        - return Generator[dict , None, None]:
        {
            "text": str,
            "audio_wav": torch.Tensor
        }
        """
        # process prompt
        message = session.ctx.state.get("message", "")
        assert message is not None

        chunk_size_list = kwargs.get("chunk_size_list", self.chunk_size_list)
        mode = kwargs.get("mode", None)  # None | "luke" for chat
        do_sample = kwargs.get("do_sample", False)  # bool
        audio_path = kwargs.get("audio_path", None)  # None | str
        prompt_audio_path = kwargs.get("prompt_audio_path", None)  # None | str
        mtp_inference_mode = kwargs.get("mtp_inference_mode", None)  # None | List[int]

        prompt_speech_feat = torch.zeros(1, 0, 80).to(self._model.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self._model.device)

        this_uuid = session.ctx.client_id
        tts_mels = []
        prev_mel = None

        is_finalize = False
        chunk_size_idx = 0
        chunk_size = chunk_size_list[chunk_size_idx]

        times = []
        start_time = time.perf_counter()

        raw_text = ""
        audio_decode_time = []
        audio_chunk = []
        for new_text in self.run_infer_stream(
            audio_path=audio_path,
            prompt_audio_path=prompt_audio_path,
            message=message,
            mode=mode,
            do_sample=do_sample,
            mtp_inference_mode=mtp_inference_mode,
        ):
            times.append(time.perf_counter() - start_time)
            # print(new_text, end="", flush=True)
            if "<|begin_of_audio|>" in new_text:
                new_text = new_text.replace("<|begin_of_audio|>", "")
            if "<|end_of_audio|>" in new_text:
                new_text = new_text.replace("<|end_of_audio|>", "")
            if "<|im_end|>" in new_text:
                new_text = new_text.replace("<|im_end|>", "")
                is_finalize = True
            audio_tokens = extract_token_ids_as_int(new_text)
            # print(f"\n{audio_tokens=}", flush=True)
            if not audio_tokens and is_finalize is False:
                raw_text += new_text
                yield {"text": new_text}
                continue
            audio_chunk.extend(audio_tokens)
            # print(f"{is_finalize=} {len(audio_chunk)=}")
            if len(audio_chunk) >= chunk_size or (is_finalize and audio_chunk):
                # print(f"\n{audio_chunk=}", flush=True)
                if chunk_size_idx < len(chunk_size_list) - 1:
                    chunk_size_idx += 1
                    chunk_size = chunk_size_list[chunk_size_idx]
                tts_token = torch.tensor(audio_chunk, device=self._model.device).unsqueeze(0)

                if prev_mel is not None:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                # gen waveform and mel-spectrogram feat
                start_time = time.perf_counter()
                tts_speech, tts_mel = self.audio_tokenizer.audio_decoder.token2wav(
                    tts_token,
                    uuid=this_uuid,
                    prompt_token=flow_prompt_speech_token.to(self._model.device),
                    prompt_feat=prompt_speech_feat.to(self._model.device),
                    finalize=is_finalize,
                )
                audio_decode_time.append(time.perf_counter() - start_time)
                yield {"audio_wav": tts_speech}

                prev_mel = tts_mel
                tts_mels.append(tts_mel)
                flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                audio_chunk = []
                start_time = time.perf_counter()

        logging.info(
            f"generate [{raw_text}] first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s"
        )

        logging.info(
            f"generate first audio segment cost time: {audio_decode_time[0]} s, {len(audio_decode_time)} segment cost time: {sum(audio_decode_time)} s"
        )


class TransformersManualTextVoiceVITALLM(TransformersManualTextSpeechVITALLM):
    """
    text to speech voice chat

    - text -> text + speech
    """

    TAG = "llm_transformers_manual_vita_text_voice"

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        prompt: str | torch.Tensor

        - return Generator[dict , None, None]:
        {
            "text": str,
            "audio": torch.Tensor
        }
        """
        assert kwargs.get("mode")

        return super().generate(session, **kwargs)


class TransformersManualVoiceVITALLM(TransformersManualTextSpeechVITALLM):
    """
    speech to speech voice chat

    - speech -> text + speech
    """

    TAG = "llm_transformers_manual_vita_voice"

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        prompt: str | torch.Tensor

        - return Generator[dict , None, None]:
        {
            "text": str,
            "audio": torch.Tensor
        }
        """
        assert kwargs.get("mode")
        assert kwargs.get("audio_path")
        # print(f"{kwargs.get('audio_path')=}", flush=True)

        return super().generate(session, **kwargs)
