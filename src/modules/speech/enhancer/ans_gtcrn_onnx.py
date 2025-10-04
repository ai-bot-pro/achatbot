# Grouped Temporal Convolutional Recurrent Network (GTCRN)
import logging
import os
import time

import numpy as np
import onnxruntime
from librosa import istft, stft

from src.common.utils.audio_resample import resample_bytes2bytes, resample_numpy2bytes
from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.session import Session, SessionCtx
from src.common.types import MODELS_DIR
from src.common.factory import EngineClass
from src.common.interface import ISpeechEnhancer


class GTCRNSpeechEnhancer(EngineClass, ISpeechEnhancer):
    """
    onnx realtime
    - https://github.com/Xiaobin-Rong/gtcrn
    - wget https://github.com/Xiaobin-Rong/gtcrn/raw/refs/heads/main/stream/onnx_models/gtcrn_simple.onnx -O ./models/gtcrn_simple.onnx
    """

    TAG = "enhancer_ans_gtcrn_onnx"
    SAMPLE_RATE = 16000

    def __init__(self, **kwargs):
        super().__init__()
        model = kwargs.get("model", os.path.join(MODELS_DIR, "gtcrn_simple.onnx"))

        self.reset()
        self.warmup_cn = kwargs.get("warmup_cn", 1)
        self.infer_session = onnxruntime.InferenceSession(
            model, None, providers=["CPUExecutionProvider"]
        )

    def warmup(self, session: Session, **kwargs):
        if self.warmup_cn <= 0:
            return
        sample_rate = kwargs.get("sample_rate") or session.ctx.sampling_rate
        assert sample_rate, "sample_rate is None or 0"

        start = time.time()
        x = bytes2NpArrayWith16(
            b" " * (sample_rate * 2 // 100),  # 10ms
        )
        x = stft(
            x,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=np.hanning(512) ** 0.5,
            center=True,
            pad_mode="constant",
            dtype=None,
        )

        inputs = np.stack([x.real, x.imag], axis=0)[None, ...].astype("float32")
        inputs = np.transpose(inputs, (0, 2, 3, 1))  # (1,F,Time,2)

        conv_cache = np.zeros([2, 1, 16, 16, 33], dtype="float32")
        tra_cache = np.zeros([2, 3, 1, 1, 16], dtype="float32")
        inter_cache = np.zeros([2, 1, 33, 16], dtype="float32")
        outputs = []
        for i in range(inputs.shape[-2]):
            out_i, conv_cache, tra_cache, inter_cache = self.infer_session.run(
                [],
                {
                    "mix": inputs[..., i : i + 1, :],
                    "conv_cache": conv_cache,
                    "tra_cache": tra_cache,
                    "inter_cache": inter_cache,
                },
            )
            outputs.append(out_i)

        outputs = np.concatenate(outputs, axis=2)
        _ = istft(
            outputs[..., 0] + 1j * outputs[..., 1],
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=np.hanning(512) ** 0.5,
        )

        logging.info(f"{self.TAG} warmup cost: {time.time() - start} s")

    def enhance(self, session: Session, **kwargs):
        sample_rate = kwargs.get("sample_rate") or session.ctx.sampling_rate
        assert sample_rate, "sample_rate is None or 0"
        audio_chunk = session.ctx.state.get("audio_chunk", b"")
        is_last = session.ctx.state.get("is_last", False)

        if len(audio_chunk) == 0:
            return audio_chunk
        if sample_rate != self.SAMPLE_RATE:
            audio_chunk = resample_bytes2bytes(
                audio_chunk,
                orig_freq=sample_rate,
                new_freq=self.SAMPLE_RATE,
            )

        x = bytes2NpArrayWith16(audio_chunk)
        x = stft(
            x,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=np.hanning(512) ** 0.5,
            center=True,
            pad_mode="constant",
            dtype=None,
        )

        inputs = np.stack([x.real, x.imag], axis=0)[None, ...].astype("float32")
        inputs = np.transpose(inputs, (0, 2, 3, 1))  # (1,F,Time,2)

        outputs = []
        for i in range(inputs.shape[-2]):
            out_i, self.conv_cache, self.tra_cache, self.inter_cache = self.infer_session.run(
                [],
                {
                    "mix": inputs[..., i : i + 1, :],
                    "conv_cache": self.conv_cache,
                    "tra_cache": self.tra_cache,
                    "inter_cache": self.inter_cache,
                },
            )
            outputs.append(out_i)

        outputs = np.concatenate(outputs, axis=2)
        out_pcm = istft(
            outputs[..., 0] + 1j * outputs[..., 1],
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=np.hanning(512) ** 0.5,
        )
        out_pcm_bytes = b""
        if len(out_pcm) > 0:
            out_pcm_bytes = resample_numpy2bytes(
                out_pcm,
                orig_freq=self.SAMPLE_RATE,
                new_freq=sample_rate,
            )
        if is_last is True:
            self.reset()

        return out_pcm_bytes

    def reset(self):
        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype="float32")
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype="float32")
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype="float32")
