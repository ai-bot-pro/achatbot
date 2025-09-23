import os
import time
import logging

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

from src.modules.speech.help.audio_mock import generate_white_noise
from src.common.utils.audio_resample import resample_bytes2bytes
from src.common.session import Session
from src.common.factory import EngineClass
from src.common.interface import ISpeechEnhancer
from src.common.types import MODELS_DIR


class DFSMNSpeechEnhancer(EngineClass, ISpeechEnhancer):
    """
    pytorch realtime
    - https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal
    - modelscope download --local_dir ./models/iic/speech_dfsmn_ans_psm_48k_causal iic/speech_dfsmn_ans_psm_48k_causal
    """

    TAG = "enhancer_ans_dfsmn"
    SAMPLE_RATE = 48000

    def __init__(self, **kwargs):
        super().__init__()
        model = kwargs.get(
            "model",
            os.path.join(
                MODELS_DIR, "iic/speech_dfsmn_ans_psm_48k_causal"
            ),  # output 48K auido sample rate
        )

        self.ans = pipeline(
            Tasks.acoustic_noise_suppression,
            model=model,
            stream_mode=True,
        )
        self.warmup_cn = kwargs.get("warmup_cn", 1)

    def warmup(self, session: Session, **kwargs):
        if self.warmup_cn <= 0:
            return
        sample_rate = kwargs.get("sample_rate") or session.ctx.sampling_rate
        assert sample_rate, "sample_rate is None or 0"

        start = time.time()
        white_noise = generate_white_noise(duration=0.1, sr=sample_rate)
        audio_data = white_noise.astype(np.int16).tobytes()
        audio_bytes_48k = resample_bytes2bytes(
            audio_data,
            orig_freq=sample_rate,
            new_freq=self.SAMPLE_RATE,
        )
        self.ans(audio_bytes_48k)
        logging.info(f"{self.TAG} warmup cost: {time.time() - start} s")

    def enhance(self, session: Session, **kwargs):
        sample_rate = kwargs.get("sample_rate") or session.ctx.sampling_rate
        assert sample_rate, "sample_rate is None or 0"
        audio_chunk = session.ctx.state.get("audio_chunk", b"")
        # is_last = session.ctx.state.get("is_last", False)

        if len(audio_chunk) == 0:
            return audio_chunk
        if sample_rate != self.SAMPLE_RATE:
            audio_chunk = resample_bytes2bytes(
                audio_chunk,
                orig_freq=sample_rate,
                new_freq=self.SAMPLE_RATE,
            )
        result = self.ans(audio_chunk)
        out_pcm = result.get(OutputKeys.OUTPUT_PCM, b"")
        if len(out_pcm) > 0:
            out_pcm = resample_bytes2bytes(
                out_pcm,
                orig_freq=self.SAMPLE_RATE,
                new_freq=sample_rate,
            )
        return out_pcm

    def reset(self):
        pass
