import asyncio
import os

import librosa
import soundfile as sf

from src.common.time_utils import to_timestamp
from src.modules.speech.asr_live import ASRLiveEnvInit
from src.common.session import Session, SessionCtx
from src.common.types import ASSETS_DIR

engine = ASRLiveEnvInit.initEngine(textnorm=bool(os.getenv("TEXTNORM", "")))
session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)

wav_path = os.path.join(ASSETS_DIR, "Chinese_prompt.wav")
samples, sr = sf.read(wav_path)
print(wav_path, samples.shape, sr)
target_sample_rate = 16000
if sr != target_sample_rate:
    samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sample_rate)
    sr = target_sample_rate
    print(f"Resampled to {target_sample_rate}Hz {samples.shape=}")
# samples = (samples * 32768).tolist() * 1
# samples = (samples * 32768) * 1


async def run(samples, sr):
    pre_len = 0
    step = int(0.1 * sr)
    start_times = []
    result = {}
    for i in range(0, len(samples), step):
        session.ctx.state["audio_chunk"] = samples[i : i + step]
        session.ctx.state["is_last"] = i + step >= len(samples)
        async for res in engine.streaming_transcribe(session):
            print(res)
            for timestamp in res["timestamps"][pre_len:]:
                start_times.append(to_timestamp(timestamp, msec=1))
            result = res
            result["start_times"] = start_times
            pre_len = len(res["timestamps"])
    print(result)


asyncio.run(run(samples, sr))

"""
python -m src.modules.speech.asr_live
TEXTNORM=1 python -m src.modules.speech.asr_live
"""
