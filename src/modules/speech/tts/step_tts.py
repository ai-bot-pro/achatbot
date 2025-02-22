import json
import logging
import math
from pathlib import Path
import re
import sys
from threading import Lock
from typing import AsyncGenerator
import os
import io

import numpy as np
import torch
import torchaudio


try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../StepAudio"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/StepAudio"))
    from src.core.llm.transformers.manual_speech_step import TransformersManualSpeechStep
    from deps.StepAudio.tokenizer import StepAudioTokenizer
    from deps.StepAudio.cosyvoice.cli.cosyvoice import CosyVoice
except ModuleNotFoundError as e:
    logging.error("In order to use step-tts, you need to `pip install achatbot[tts_step]`.")
    raise Exception(f"Missing module: {e}")

from src.common.random import set_all_random_seed
from src.common.types import PYAUDIO_PAFLOAT32, ASSETS_DIR
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.step import StepTTSArgs
from src.types.llm.transformers import TransformersSpeechLMArgs
from .base import BaseTTS


class StepTTS(BaseTTS, ITts):
    """
    support tts mode:
    - lm_gen: text+ref audio waveform lm gen audio wav code to gen waveform with static batch stream:
        text+ref audio waveform -> tokenizer -> text+audio token ids -> step1 lm  -> audio token ids (wav_code) -> flow(CFM) -> mel - vocoder(hifi) -> waveform
    - voice_clone: voice clone w/o lm gen, decode wav code:
        src+ref audio waveform -> speech tokenizer-> audio token ids (wav_code) -> flow(CFM) -> mel - vocoder(hifi) -> clone ref audio waveform
    """

    TAG = "tts_step"

    def __init__(self, **kwargs) -> None:
        self.args = StepTTSArgs(**kwargs)
        assert (
            self.args.stream_factor >= 2
        ), "stream_factor must >=2 increase for better speech quality, but rtf slow (speech quality vs rtf)"

        self.encoder = StepAudioTokenizer(self.args.speech_tokenizer_model_path)

        self.lm_args = TransformersSpeechLMArgs(**self.args.lm_args)
        self.lm_model = TransformersManualSpeechStep(**self.lm_args.__dict__)
        # session ctx dict with lock, maybe need a session class
        self.session_lm_generat_lock = Lock()
        self.session_lm_generated_ids = {}  # session_id: ids(ptr)

        self.common_cosy_model = CosyVoice(
            os.path.join(self.lm_args.lm_model_name_or_path, "CosyVoice-300M-25Hz")
        )
        self.music_cosy_model = CosyVoice(
            os.path.join(self.lm_args.lm_model_name_or_path, "CosyVoice-300M-25Hz-Music")
        )

        self.sys_prompt_dict = {
            "sys_prompt_for_rap": "请参考对话历史里的音色，用RAP方式将文本内容大声说唱出来。",
            "sys_prompt_for_vocal": "请参考对话历史里的音色，用哼唱的方式将文本内容大声唱出来。",
            "sys_prompt_wo_spk": '作为一名卓越的声优演员，你的任务是根据文本中（）或()括号内标注的情感、语种或方言、音乐哼唱、语音调整等标签，以丰富细腻的情感和自然顺畅的语调来朗读文本。\n# 情感标签涵盖了多种情绪状态，包括但不限于：\n- "高兴1"\n- "高兴2"\n- "生气1"\n- "生气2"\n- "悲伤1"\n- "撒娇1"\n\n# 语种或方言标签包含多种语言或方言，包括但不限于：\n- "中文"\n- "英文"\n- "韩语"\n- "日语"\n- "四川话"\n- "粤语"\n- "广东话"\n\n# 音乐哼唱标签包含多种类型歌曲哼唱，包括但不限于：\n- "RAP"\n- "哼唱"\n\n# 语音调整标签，包括但不限于：\n- "慢速1"\n- "慢速2"\n- "快速1"\n- "快速2"\n\n请在朗读时，根据这些情感标签的指示，调整你的情感、语气、语调和哼唱节奏，以确保文本的情感和意义得到准确而生动的传达，如果没有()或（）括号，则根据文本语义内容自由演绎。',
            "sys_prompt_with_spk": '作为一名卓越的声优演员，你的任务是根据文本中（）或()括号内标注的情感、语种或方言、音乐哼唱、语音调整等标签，以丰富细腻的情感和自然顺畅的语调来朗读文本。\n# 情感标签涵盖了多种情绪状态，包括但不限于：\n- "高兴1"\n- "高兴2"\n- "生气1"\n- "生气2"\n- "悲伤1"\n- "撒娇1"\n\n# 语种或方言标签包含多种语言或方言，包括但不限于：\n- "中文"\n- "英文"\n- "韩语"\n- "日语"\n- "四川话"\n- "粤语"\n- "广东话"\n\n# 音乐哼唱标签包含多种类型歌曲哼唱，包括但不限于：\n- "RAP"\n- "哼唱"\n\n# 语音调整标签，包括但不限于：\n- "慢速1"\n- "慢速2"\n- "快速1"\n- "快速2"\n\n请在朗读时，使用[{}]的声音，根据这些情感标签的指示，调整你的情感、语气、语调和哼唱节奏，以确保文本的情感和意义得到准确而生动的传达，如果没有()或（）括号，则根据文本语义内容自由演绎。',
        }

        self.speakers_info = {}
        self.register_speakers()

        # lm model gen warmup, codec model decode(flow + hifi) don't to warmup

    def register_speakers(self):
        self.speakers_info = {}

        speackers_info_path = os.path.join(ASSETS_DIR, "speakers/speakers_info.json")
        with open(speackers_info_path, "r") as f:
            speakers_info = json.load(f)

        for speaker_id, prompt_text in speakers_info.items():
            prompt_wav_path = os.path.join(ASSETS_DIR, f"speakers/{speaker_id}_prompt.wav")
            (
                ref_audio_code,
                ref_audio_token,
                ref_audio_token_len,
                ref_speech_feat,
                ref_speech_feat_len,
                ref_speech_embedding,
            ) = self.preprocess_prompt_wav(prompt_wav_path)

            self.speakers_info[speaker_id] = {
                "ref_text": prompt_text,
                "ref_audio_code": ref_audio_code,
                "ref_speech_feat": ref_speech_feat.to(torch.bfloat16),
                "ref_speech_feat_len": ref_speech_feat_len,
                "ref_speech_embedding": ref_speech_embedding.to(torch.bfloat16),
                "ref_audio_token": ref_audio_token,
                "ref_audio_token_len": ref_audio_token_len,
            }
            logging.info(f"Registered speaker: {speaker_id}")

    def wav2code(self, prompt_wav_path: str):
        prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)  # multi-channel to mono
        prompt_code, _, _ = self.encoder.wav2token(prompt_wav, prompt_wav_sr)
        return prompt_code

    def preprocess_prompt_wav(self, prompt_wav_path: str):
        prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)  # multi-channel to mono
        prompt_wav_16k = torchaudio.transforms.Resample(orig_freq=prompt_wav_sr, new_freq=16000)(
            prompt_wav
        )
        prompt_wav_22k = torchaudio.transforms.Resample(orig_freq=prompt_wav_sr, new_freq=22050)(
            prompt_wav
        )

        speech_feat, speech_feat_len = self.common_cosy_model.frontend._extract_speech_feat(
            prompt_wav_22k
        )
        speech_embedding = self.common_cosy_model.frontend._extract_spk_embedding(prompt_wav_16k)

        prompt_code, _, _ = self.encoder.wav2token(prompt_wav, prompt_wav_sr)
        prompt_token = torch.tensor([prompt_code], dtype=torch.long) - 65536
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long)

        return (
            prompt_code,
            prompt_token,
            prompt_token_len,
            speech_feat,
            speech_feat_len,
            speech_embedding,
        )

    def set_voice(self, ref_audio_path: str, **kwargs):
        """
        - save to speacker info dict
        TODO: save dict to dist kv store
        """
        assert os.path.exists(ref_audio_path), "ref_audio_path is not exists"
        assert kwargs.get(
            "ref_speaker", None
        ), "ref_speaker is not exists"  # maybe use random speaker
        assert kwargs.get(
            "ref_text", None
        ), "ref_text is not exists"  # maybe use asr to get ref_text
        ref_speaker = kwargs.get("ref_speaker")
        ref_text = kwargs.get("ref_text")

        (
            ref_audio_code,
            ref_audio_token,
            ref_audio_token_len,
            ref_speech_feat,
            ref_speech_feat_len,
            ref_speech_embedding,
        ) = self.preprocess_prompt_wav(ref_audio_path)

        self.speakers_info[ref_speaker] = {
            "ref_text": ref_text,
            "ref_audio_code": ref_audio_code,
            "ref_speech_feat": ref_speech_feat.to(torch.bfloat16),
            "ref_speech_feat_len": ref_speech_feat_len,
            "ref_speech_embedding": ref_speech_embedding.to(torch.bfloat16),
            "ref_audio_token": ref_audio_token,
            "ref_audio_token_len": ref_audio_token_len,
        }

    def get_voices(self) -> list:
        return list(self.speakers_info.keys())

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": 22050,
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    def detect_instruction_name(self, text):
        instruction_name = ""
        match_group = re.match(r"^([（\(][^\(\)()]*[）\)]).*$", text, re.DOTALL)
        if match_group is not None:
            instruction = match_group.group(1)
            instruction_name = instruction.strip("()（）")
        return instruction_name

    def set_system_prompt(self, text, ref_speaker: str = "Tingting"):
        sys_prompt = self.sys_prompt_dict["sys_prompt_wo_spk"]
        instruction_name = self.detect_instruction_name(text)
        if instruction_name:
            if "哼唱" in text:
                sys_prompt = self.sys_prompt_dict["sys_prompt_for_vocal"]
            else:
                sys_prompt = self.sys_prompt_dict["sys_prompt_for_rap"]
        elif ref_speaker:
            sys_prompt = self.sys_prompt_dict["sys_prompt_with_spk"].format(ref_speaker)
        self.lm_model.set_system_prompt(sys_prompt=sys_prompt)

    def voice_clone(self, session: Session, ref_speaker: str, cosy_model, **kwargs):
        """
        - voice_clone: voice clone w/o lm gen, decode wav code:
        src+ref audio waveform -> speech tokenizer-> audio token ids (wav_code) -> flow(CFM) -> mel - vocoder(hifi) -> clone ref audio waveform
        """
        src_audio_path = kwargs.get("src_audio_path", None)
        if not src_audio_path or not os.path.exists(src_audio_path):
            logging.error(f"{src_audio_path} is not exists")
            return None

        src_audio_code = self.wav2code(src_audio_path)
        tensor_audio_token_ids = torch.tensor([src_audio_code]).to(torch.long).to("cuda") - 65536
        tts_speech = cosy_model.token_to_wav_offline(
            tensor_audio_token_ids,
            self.speakers_info[ref_speaker]["ref_speech_feat"].to(torch.bfloat16),
            self.speakers_info[ref_speaker]["ref_speech_feat_len"],
            self.speakers_info[ref_speaker]["ref_audio_token"],
            self.speakers_info[ref_speaker]["ref_audio_token_len"],
            self.speakers_info[ref_speaker]["ref_speech_embedding"].to(torch.bfloat16),
        )
        return tts_speech.float().numpy().tobytes()

    async def lm_gen(
        self,
        session: Session,
        text: str,
        ref_speaker: str,
        batch_size: int,
        cosy_model,
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """
        - lm_gen: text+ref audio waveform lm gen audio wav code to gen waveform with static batch stream:
        text+ref audio waveform -> tokenizer -> text+audio token ids -> step1 lm -> audio token ids (wav_code) -> flow(CFM) -> mel - vocoder(hifi) -> waveform
        """
        session_id = session.ctx.client_id

        self.set_system_prompt(text, ref_speaker=ref_speaker)

        one_shot_ref_text = self.speakers_info[ref_speaker]["ref_text"]
        one_shot_ref_audio = self._tokenizer.decode(
            self.speakers_info[ref_speaker]["ref_audio_code"]
        )
        prompt = f"<s><|BOT|><s> system\n{self.sys_prompt}"
        prompt += f"<|EOT|><|BOT|><s> human\n{one_shot_ref_text}" if one_shot_ref_text else ""
        prompt += f"<|EOT|><|BOT|><s> assistant\n{one_shot_ref_audio}" if one_shot_ref_audio else ""
        prompt += f"<|EOT|><|BOT|><s> human\n{text}"
        prompt += "<|EOT|><|BOT|><s> assistant\n"

        session.ctx.state["prompt"] = prompt
        audio_vq_tokens = self.lm_model.generate(session, **kwargs)
        for token_id in audio_vq_tokens:
            # print(token_id, end=",", flush=True)
            if token_id == self.lm_model.end_token_id:  # skip <|EOT|>, break
                break
            self.session_lm_generated_ids[session_id].append(token_id)
            if len(self.session_lm_generated_ids[session_id]) % batch_size == 0:
                batch = (
                    torch.tensor(self.session_lm_generated_ids[session_id])
                    .unsqueeze(0)
                    .to(cosy_model.model.device)
                    - 65536
                )  # [T] -> [1,T]
                logging.debug(f"batch: {batch}")
                # Process each batch
                sub_tts_speech = cosy_model.token_to_wav_offline(
                    batch,
                    self.speakers_info[ref_speaker]["ref_speech_feat"].to(torch.bfloat16),
                    self.speakers_info[ref_speaker]["ref_speech_feat_len"],
                    self.speakers_info[ref_speaker]["ref_audio_token"],
                    self.speakers_info[ref_speaker]["ref_audio_token_len"],
                    self.speakers_info[ref_speaker]["ref_speech_embedding"].to(torch.bfloat16),
                )
                yield sub_tts_speech.float().numpy().tobytes()
                with self.session_lm_generat_lock:
                    self.session_lm_generated_ids[session_id] = []

        if len(self.session_lm_generated_ids[session_id]) > 0:
            batch = (
                torch.tensor(self.session_lm_generated_ids[session_id])
                .unsqueeze(0)
                .to(cosy_model.model.device)
                - 65536
            )  # [T] -> [1,T]
            logging.debug(f"batch: {batch}")
            # Process each batch
            sub_tts_speech = cosy_model.token_to_wav_offline(
                batch,
                self.speakers_info[ref_speaker]["ref_speech_feat"].to(torch.bfloat16),
                self.speakers_info[ref_speaker]["ref_speech_feat_len"],
                self.speakers_info[ref_speaker]["ref_audio_token"],
                self.speakers_info[ref_speaker]["ref_audio_token_len"],
                self.speakers_info[ref_speaker]["ref_speech_embedding"].to(torch.bfloat16),
            )
            yield sub_tts_speech.float().numpy().tobytes()

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if "cuda" in str(self.lm_model._model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.lm_args.lm_gen_seed)
        set_all_random_seed(seed)

        ref_speaker = kwargs.pop("ref_speaker", "Tingting")
        instruction_name = self.detect_instruction_name(text)
        cosy_model = self.common_cosy_model
        if instruction_name in ["RAP", "哼唱"]:
            cosy_model = self.music_cosy_model
            ref_speaker = f"{ref_speaker}{instruction_name}"
            if ref_speaker not in self.speakers_info:
                ref_speaker = f"Tingting{instruction_name}"
        if ref_speaker and ref_speaker not in self.speakers_info:
            ref_speaker = "Tingting"
        logging.debug(f"use ref_speaker: {ref_speaker}")

        assert (
            kwargs.get("stream_factor", self.args.stream_factor) >= 2
        ), "stream_factor must >=2 increase for better speech quality, but rtf slow (speech quality vs rtf)"
        batch_size = math.ceil(
            kwargs.get("stream_factor", self.args.stream_factor)
            * cosy_model.model.flow.input_frame_rate
        )

        session_id = session.ctx.client_id
        with self.session_lm_generat_lock:
            self.session_lm_generated_ids[session_id] = []

        tts_mode = kwargs.get("tts_mode", self.args.tts_mode)
        if tts_mode == "voice_clone":
            tts_speech = self.voice_clone(session, ref_speaker, cosy_model, **kwargs)
            yield tts_speech
        else:  # lm_gen
            async for item in self.lm_gen(
                session, text, ref_speaker, batch_size, cosy_model, **kwargs
            ):
                yield item

        with self.session_lm_generat_lock:
            self.session_lm_generated_ids.pop(session_id)
        torch.cuda.empty_cache()
