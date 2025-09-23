import os
import time
import logging

from pyrnnoise import RNNoise

from src.common.utils.audio_resample import resample_bytes2bytes, resample_bytes2numpy
from src.common.session import Session
from src.common.factory import EngineClass
from src.common.interface import ISpeechEnhancer


class RNNoiseSpeechEnhancer(EngineClass, ISpeechEnhancer):
    """
    realtime ANS
    - ⭐️ https://jmvalin.ca/demo/rnnoise/
    - ⭐️ A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement](https://arxiv.org/abs/1709.08243)
    - https://gitlab.xiph.org/xiph/rnnoise or https://github.com/xiph/rnnoise.git
    - https://github.com/werman/noise-suppression-for-voice (rnnoise c++ binding)
    - https://github.com/pengzhendong/pyrnnoise (rnnoise python ctypes binding)

    # NOTE:
    # - rnnoise min support 10ms for 48000 sample rate
    # - if not 10ms frame size, AudioGraph to buffering until 10ms frame size
    # SEE:
    # - https://av.basswood-io.com/docs/stable/api/filter.html#bv.filter.graph.Graph
    # https://ffmpeg.org/ffmpeg-filters.html#abuffer

    # in_audio_path = "./records/speech_with_noise_48k.wav"
    # frame_bytes_size = 480 * 2  # 10ms
    # frame_bytes_size = 48 * 2  # 1ms
    """

    TAG = "enhancer_ans_rnnoise"
    SAMPLE_RATE = 48000

    def __init__(self, **kwargs):
        super().__init__()
        self.denoiser = RNNoise(sample_rate=self.SAMPLE_RATE)
        self.warmup_cn = kwargs.get("warmup_cn", 1)

    def warmup(self, session: Session, **kwargs):
        if self.warmup_cn <= 0:
            return
        sample_rate = kwargs.get("sample_rate") or session.ctx.sampling_rate
        assert sample_rate, "sample_rate is None or 0"

        start = time.time()
        waveform_48k_int16 = resample_bytes2numpy(
            b" " * (sample_rate * 2 // 100),  # 10ms
            orig_freq=sample_rate,
            new_freq=self.SAMPLE_RATE,
        )
        self.denoiser.denoise_chunk(waveform_48k_int16, False)
        logging.info(f"{self.TAG} warmup cost: {time.time() - start} s")

    def enhance(self, session: Session, **kwargs):
        sample_rate = kwargs.get("sample_rate") or session.ctx.sampling_rate
        assert sample_rate, "sample_rate is None or 0"
        audio_chunk = session.ctx.state.get("audio_chunk", b"")
        is_last = session.ctx.state.get("is_last", False)
        if len(audio_chunk) == 0:
            return audio_chunk

        audio_chunk_np = resample_bytes2numpy(
            audio_chunk,
            orig_freq=sample_rate,
            new_freq=self.SAMPLE_RATE,
        )

        out_pcm = b""
        for _, frame in self.denoiser.denoise_chunk(audio_chunk_np, is_last):
            out_pcm += frame.tobytes()
        if len(out_pcm) > 0:
            out_pcm = resample_bytes2bytes(
                out_pcm,
                orig_freq=self.SAMPLE_RATE,
                new_freq=sample_rate,
            )
        if is_last is True:
            self.denoiser.reset()

        return out_pcm

    def reset(self):
        self.denoiser.reset()
