import os
import time
import wave
import logging

from src.modules.speech.enhancer import SpeechEnhancerEnvInit
from src.common.session import Session, SessionCtx
from src.common.logger import Logger

Logger.init(logging.INFO)

engine = SpeechEnhancerEnvInit.initEngine()
session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)
engine.warmup(session)

in_audio_path = "./records/speech_with_noise16k.wav"
# block_bytes_size = 160 * 2
# block_bytes_size = 16 * 2
block_bytes_size = 512 * 2  # silero vad frame size * 2


for i in range(10):
    # engine.reset()
    audio_pcm = b""
    begin = time.time()
    with open(os.path.join(os.getcwd(), in_audio_path), "rb") as f:
        # 跳过 WAV 头
        f.read(44)

        audio_bytes = f.read(block_bytes_size)
        while len(audio_bytes) > 0:
            start = time.time()
            print("before", len(audio_bytes))
            session.ctx.state["audio_chunk"] = audio_bytes
            session.ctx.state["sample_rate"] = 16000
            session.ctx.state["is_last"] = len(audio_bytes) != block_bytes_size
            out_pcm = engine.enhance(session)
            print("after", len(out_pcm))
            audio_pcm += out_pcm
            print(f"cost: {time.time() - start} s")

            audio_bytes = f.read(block_bytes_size)

    print(f"total cost: {time.time() - begin} s")

out_path = f"./records/output_{engine.SELECTED_TAG}_stream.wav"
with wave.open(out_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(audio_pcm)
    print(f"{in_audio_path=} save to {out_path}")

"""
# use rnnoise c lib python binding
SPEECH_ENHANCER_TAG=enhancer_ans_rnnoise python -m src.modules.speech.enhancer

# pytorch dfsmn (good but slow)
SPEECH_ENHANCER_TAG=enhancer_ans_dfsmn python -m src.modules.speech.enhancer

# onnx gtcrn (need PT with scene datasets)
SPEECH_ENHANCER_TAG=enhancer_ans_gtcrn_onnx python -m src.modules.speech.enhancer
"""
