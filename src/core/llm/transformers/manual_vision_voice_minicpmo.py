import logging
import os
from threading import Thread
import uuid

import numpy as np

try:
    import torch
    import librosa
    import soundfile as sf
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    # from auto_gptq import AutoGPTQForCausalLM
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Qwen2-VL, you need to `pip install achatbot[llm_transformers_manual_vision_voice_minicpmo]`,"
    )
    raise Exception(f"Missing module: {e}")

from src.common.random import set_all_random_seed
from src.common.utils.helper import get_device, print_model_params
from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.types.llm.transformers import TransformersLMArgs
from src.common.types import RECORDS_DIR
from .base import TransformersBaseLLM


class TransformersManualVisionVoiceMiniCPMO(TransformersBaseLLM):
    TAG = "llm_transformers_manual_vision_voice_minicpmo"

    def __init__(self, **args) -> None:
        # session sys settings
        # language
        self.language = args.pop("language", "zh")
        # interation mode
        # "default": default system prompt and not refer to any task
        # "omni": input video and audio simultaneously
        # "audio_assistant": Default voice-only mode, the model will use the ref_audio's voice to reply user's question as a helpful assistant.
        # "audio_roleplay": Roleplay voice-only mode, the model will use the ref_audio's voice to reply, and also role-play the character based on the audio prompt.
        # "voice_cloning": TTS mode, the model will clone the voice of ref_audio.
        self.interaction_mode = args.pop("interaction_mode", "omni")
        # reference audio
        ref_audio_path = args.pop("ref_audio_path", None)
        self.ref_audio = None
        if ref_audio_path is not None:
            self.ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

        # init vision/voice
        init_vision = args.pop("init_vision", True)
        init_audio = args.pop("init_audio", True)
        init_tts = args.pop("init_tts", True)

        # whether gen result audio (tts)
        self.generate_audio = args.pop("generate_audio", True)
        # whether save result audio
        self.save_output = args.pop("save_oupt", False)

        self.args = TransformersLMArgs(**args)
        if self.args.lm_torch_dtype != "auto":
            self.torch_dtype = getattr(torch, self.args.lm_torch_dtype)
        else:
            self.torch_dtype = "auto"

        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)

        # load omni model default, the default init_vision/init_audio/init_tts is True

        # if load vision-only model, please set init_audio=False and init_tts=False
        # if load audio-only model, please set init_vision=False
        model = AutoModel.from_pretrained(
            self.args.lm_model_name_or_path,
            trust_remote_code=True,
            attn_implementation=self.args.lm_attn_impl,
            torch_dtype=self.torch_dtype,
            init_vision=init_vision,
            init_audio=init_audio,
            init_tts=init_tts,
        )
        print_model_params(model, self.TAG)
        self._model = model.eval().cuda()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True
        )

        # In addition to vision-only mode, tts processor and vocos also needs to be initialized
        if init_audio is False and init_tts is False:
            model.init_tts()

        self._sys_msg = self._model.get_sys_prompt(
            ref_audio=self.ref_audio, mode=self.interaction_mode, language=self.language
        )

        self._chat_history = ChatHistory(self.args.chat_history_size)
        if self._sys_msg:
            self._chat_history.init(self._sys_msg)

        self.warmup()

    def set_system_prompt(self, **kwargs):
        # session sys settings
        # language
        self.language = kwargs.pop("language", self.language)
        # interation mode
        # "default": default system prompt and not refer to any task
        # "omni": input video and audio simultaneously
        # "audio_assistant": Default voice-only mode, the model will use the ref_audio's voice to reply user's question as a helpful assistant.
        # "audio_roleplay": Roleplay voice-only mode, the model will use the ref_audio's voice to reply, and also role-play the character based on the audio prompt.
        # "voice_cloning": TTS mode, the model will clone the voice of ref_audio.
        self.interaction_mode = kwargs.pop("interaction_mode", self.interaction_mode)
        # reference audio
        ref_audio_path = kwargs.pop("ref_audio_path", None)
        if ref_audio_path is not None:
            self.ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

        self._sys_msg = self._model.get_sys_prompt(
            ref_audio=self.ref_audio, mode=self.interaction_mode, language=self.language
        )

    def warmup(self):
        dummy_pil_images = [Image.new("RGB", (100, 100), color="white")]
        dummy_input_text = self.args.warnup_prompt
        msgs = [
            {
                "role": "user",
                "content": [dummy_pil_images, dummy_input_text],
            }
        ]
        for i in self.args.warnup_steps:
            answer = self._model.chat(msgs=msgs, tokenizer=self._tokenizer)
            logging.debug(f"warmup {i}->answer:{answer}")

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        logging.debug(f"kwargs: {kwargs}")
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = ""
        if isinstance(session.ctx.state["prompt"], list):
            prompt = session.ctx.state["prompt"]

        message = {"role": self.args.user_role, "content": prompt}
        self._chat_history.append(message)
        msgs = self._chat_history.to_list()

        # prefill system prompt and msgs and decode first token (prefill_decode(decode first token for TTFT(Time to First Token)))
        # https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1049
        self._model.streaming_prefill(
            session_id=session.ctx.client_id, msgs=[self._sys_msg, msgs], tokenizer=self._tokenizer
        )

        # generate(decoding) (decode_n_tokens for TPOT(Time per Output Token))
        # https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1168
        lm_gen_temperature = kwargs.get("temperature", self.args.lm_gen_temperature)
        generate_audio = kwargs.get("generate_audio", self.generate_audio)
        streamer = self._model.streaming_generate(
            session_id=session.ctx.client_id,
            tokenizer=self._tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", self.args.lm_gen_max_new_tokens),
            min_new_tokens=kwargs.get("min_new_tokens", self.args.lm_gen_min_new_tokens),
            do_sample=False if lm_gen_temperature == 0 else True,
            temperature=lm_gen_temperature,
            top_p=kwargs.get("top_p", self.args.lm_gen_top_p),
            top_k=kwargs.get("top_k", self.args.lm_gen_top_k),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.args.lm_gen_repetition_penalty
            ),
            generate_audio=generate_audio,
        )

        audios = []
        text = ""

        if self.generate_audio:
            # https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1496
            # _generate_mel_spec_audio_streaming -> streamer (OmniOutput)
            for r in streamer:  # OmniOutput
                audio_wav = r.audio_wav
                sampling_rate = r.sampling_rate
                txt = r.text

                audios.append(audio_wav)
                text += txt
                yield r.__dict__  # OmniOutput.__dict__

            if self.save_output is True:
                res = np.concatenate(audios)
                session_dir = os.path.join(RECORDS_DIR, session.ctx.client_id)
                os.makedirs(session_dir, exist_ok=True)
                path = os.path.join(session_dir, f"{uuid.uuid4()}.wav")
                sf.write(path, res, samplerate=sampling_rate)
                logging.info(f"gen text:{text}; save to {path}")
        else:
            # https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1230
            # llm_generate_chunk -> dict {"text":text}
            for r in res:
                text += r["text"]
                yield r
            logging.info(f"gen text:{text}")
