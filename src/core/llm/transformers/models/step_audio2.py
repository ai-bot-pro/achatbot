import os
import sys
import logging
from threading import Thread
from typing import BinaryIO

try:
    import torch
    import numpy

    from transformers import GenerationConfig

    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../../StepAudio2"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../../deps/StepAudio2"))

    from deps.StepAudio2.stepaudio2 import StepAudio2, StepAudio2Base
    from deps.StepAudio2.utils import (
        compute_token_num,
        load_audio,
        log_mel_spectrogram,
        padding_mels,
    )

    from src.core.llm.transformers.streamer import TokenStreamer
    from src.common.utils.helper import get_device

except ModuleNotFoundError as e:
    raise Exception(f"Missing module: {e}")


class StepAudio2StreamBase(StepAudio2Base):
    def __init__(self, model_path: str, verbose: bool = False):
        super().__init__(model_path)
        self._verbose = verbose

    def apply_chat_template(self, messages: list):
        """
        add np.ndarray/torch.Tensor audio msg support
        - audio sample rate: 16000
        """
        results = []
        mels = []
        for msg in messages:
            content = msg
            if isinstance(content, str):
                text_with_audio = content
                results.append(text_with_audio)
            elif isinstance(content, dict):
                if content["type"] == "text":
                    results.append(f"{content['text']}")
                elif content["type"] == "audio":
                    audio = content["audio"]
                    if isinstance(audio, (BinaryIO, str, os.PathLike)):
                        audio = load_audio(audio)
                    elif isinstance(audio, numpy.ndarray):
                        audio = torch.from_numpy(audio, dtype=torch.float32)
                    assert isinstance(audio, torch.Tensor), f"Unsupported audio type: {type(audio)}"
                    if len(audio.shape) > 1:  # [1, size]
                        audio = audio.squeeze(0)  # [size]
                    for i in range(0, audio.shape[0], 16000 * 25):
                        mel = log_mel_spectrogram(
                            audio[i : i + 16000 * 25], n_mels=128, padding=479
                        )
                        mels.append(mel)
                        audio_tokens = "<audio_patch>" * compute_token_num(mel.shape[1])
                        results.append(f"<audio_start>{audio_tokens}<audio_end>")
                elif content["type"] == "token":
                    results.append(content["token"])
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        return results, mels

    def __call__(self, messages: list, **kwargs):
        messages, mels = self.apply_chat_template(messages)
        if self._verbose:
            print(f"messages: {messages}")
            if len(mels) > 0:
                print(f"{len(mels)=} {mels[0].shape=}")

        # Tokenize prompts
        prompt_ids = []
        for msg in messages:
            if isinstance(msg, str):
                prompt_ids.append(
                    self.llm_tokenizer(text=msg, return_tensors="pt", padding=True)["input_ids"]
                )
            elif isinstance(msg, list):
                prompt_ids.append(torch.tensor([msg], dtype=torch.int32))
            else:
                raise ValueError(f"Unsupported content type: {type(msg)}")
        prompt_ids = torch.cat(prompt_ids, dim=-1).cuda()
        attention_mask = torch.ones_like(prompt_ids)

        # mels = None if len(mels) == 0 else torch.stack(mels).cuda()
        # mel_lengths = None if mels is None else torch.tensor([mel.shape[1] - 2 for mel in mels], dtype=torch.int32, device='cuda')
        if len(mels) == 0:
            mels = None
            mel_lengths = None
        else:
            mels, mel_lengths = padding_mels(mels)
            mels = mels.cuda()
            mel_lengths = mel_lengths.cuda()

        generation_config = dict(
            # max_new_tokens=256,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generation_config.update(kwargs)
        generation_config = GenerationConfig(**generation_config)
        logging.debug(f"generation_config: {generation_config}")

        streamer = TokenStreamer(skip_prompt=True)

        generation_kwargs = dict(
            input_ids=prompt_ids,
            wavs=mels,
            wav_lens=mel_lengths,
            attention_mask=attention_mask,
            generation_config=generation_config,
            streamer=streamer,
            tokenizer=self.llm_tokenizer,
        )

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        stop_ids = (
            [generation_config.eos_token_id]
            if isinstance(generation_config.eos_token_id, int)
            else generation_config.eos_token_id
        )
        for token_id in streamer:
            if token_id in stop_ids:
                break
            yield token_id


class StepAudio2Stream(StepAudio2StreamBase):
    def apply_chat_template(self, messages: list):
        """
        add np.ndarray/torch.Tensor audio msg support
        - audio sample rate: 16000
        """
        results = []
        mels = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"
            if isinstance(content, str):
                text_with_audio = f"<|BOT|>{role}\n{content}"
                text_with_audio += "<|EOT|>" if msg.get("eot", True) else ""
                results.append(text_with_audio)
            elif isinstance(content, list):
                results.append(f"<|BOT|>{role}\n")
                for item in content:
                    if item["type"] == "text":
                        results.append(f"{item['text']}")
                    elif item["type"] == "audio":
                        audio = item["audio"]
                        if isinstance(audio, (BinaryIO, str, os.PathLike)):
                            audio = load_audio(audio)
                        elif isinstance(audio, numpy.ndarray):
                            audio = torch.from_numpy(audio, dtype=torch.float32)
                        assert isinstance(audio, torch.Tensor), (
                            f"Unsupported audio type: {type(audio)}"
                        )
                        if len(audio.shape) > 1:  # [1, size]
                            audio = audio.squeeze(0)  # [size]
                        for i in range(0, audio.shape[0], 16000 * 25):
                            mel = log_mel_spectrogram(
                                audio[i : i + 16000 * 25], n_mels=128, padding=479
                            )
                            mels.append(mel)
                            audio_tokens = "<audio_patch>" * compute_token_num(mel.shape[1])
                            results.append(f"<audio_start>{audio_tokens}<audio_end>")
                    elif item["type"] == "token":
                        results.append(item["token"])
                if msg.get("eot", True):
                    results.append("<|EOT|>")
            elif content is None:
                results.append(f"<|BOT|>{role}\n")
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        # print(results)
        return results, mels
