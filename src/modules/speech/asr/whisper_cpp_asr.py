import os
import logging
import asyncio
from typing import AsyncGenerator


from src.common.session import Session
from src.modules.speech.asr.base import ASRBase
from src.common.device_cuda import CUDAInfo
from src.common.time_utils import to_timestamp


class WhisperCPPAsr(ASRBase):
    """
    - https://github.com/ggml-org/whisper.cpp
    - https://huggingface.co/ggerganov/whisper.cpp
    - https://github.com/absadiki/pywhispercpp
    """

    TAG = "whisper_cpp_asr"

    def __init__(self, **args) -> None:
        super().__init__(**args)
        try:
            from pywhispercpp.model import Model

        except ImportError as e:
            print("you need to `pip install achatbot[whisper_cpp]`")
            raise Exception(f"Missing module: {e}")

        # https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.PARAMS_SCHEMA
        # https://absadiki.github.io/pywhispercpp/#pywhispercpp.model.Model
        self.model = Model(
            self.args.model_name_or_path,
            # models_dir=self.args.download_path,
            n_threads=os.cpu_count(),
            language=self.args.language,
        )
        logging.info(f"load {self.TAG} {self.model}")

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        segments = await asyncio.to_thread(
            self.model.transcribe,
            self.asr_audio.copy(),
        )
        for segment in segments:
            yield segment.text

    async def transcribe(self, session: Session) -> dict:
        # https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.PARAMS_SCHEMA
        # https://absadiki.github.io/pywhispercpp/#pywhispercpp.model.Model.transcribe
        segments = await asyncio.to_thread(
            self.model.transcribe,
            self.asr_audio.copy(),
        )

        words = []
        text = ""
        for segment in segments:
            words.append(
                {
                    "word": segment.text,
                    "start": segment.t0,
                    "end": segment.t1,
                    "start_time": to_timestamp(segment.t0),
                    "end_time": to_timestamp(segment.t1),
                }
            )
            text += segment.text

        res = {
            "language": self.args.language,
            "language_probability": None,
            "text": text,
            "words": words,
        }
        return res


class WhisperCPPCstyleAsr(ASRBase):
    """
    - https://github.com/ggml-org/whisper.cpp
    - https://huggingface.co/ggerganov/whisper.cpp
    - https://github.com/ggml-org/whisper.cpp/blob/master/include/whisper.h
    - https://github.com/ggml-org/whisper.cpp/issues/9
    -
    """

    TAG = "whisper_cpp_cstyle_asr"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        try:
            from whispercpy import WhisperCPP

        except ImportError as e:
            print("you need to `pip install achatbot[whisper_cpp_cpy]`")
            raise Exception(f"Missing module: {e}")

        info = CUDAInfo()
        library_path = os.getenv(
            "WHISPER_CPP_LIB",
            "/Users/wuyong/project/pywhispercpp/whisper.cpp/build/src/libwhisper.dylib",
        )
        self.model = WhisperCPP(
            library_path, self.args.model_name_or_path, use_gpu=info.is_cuda, verbose=True
        )

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        segments = await asyncio.to_thread(
            self.model.transcribe,
            self.asr_audio.copy(),
            token_timestamps=True,
            language=self.args.language,
        )
        for segment in segments:
            yield segment.text

    async def transcribe(self, session: Session) -> dict:
        # https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.PARAMS_SCHEMA
        # https://absadiki.github.io/pywhispercpp/#pywhispercpp.model.Model.transcribe
        # NOTE: if token_timestamps=True, language=zh, token text have special char, need prefix buff decode
        segments = await asyncio.to_thread(
            self.model.transcribe,
            self.asr_audio.copy(),
            token_timestamps=True,
            language=self.args.language,
        )

        words = []
        text = ""
        for segment in segments:
            words.append(
                {
                    "word": segment.text,
                    "start": segment.t0,
                    "end": segment.t1,
                    "start_time": to_timestamp(segment.t0),
                    "end_time": to_timestamp(segment.t1),
                }
            )
            text += segment.text

        res = {
            "language": self.args.language,
            "language_probability": None,
            "text": text,
            "words": words,
        }
        return res


"""
python -m src.modules.speech.asr.whisper_cpp_asr
"""
if __name__ == "__main__":
    from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR

    audio_file = "./test/audio_files/asr_example_zh.wav"
    # segments = asr.model.transcribe(audio_file)
    # print(segments)
    session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)

    # asr = WhisperCPPAsr(model_name_or_path="./models/ggml-base.bin")
    # asr.set_audio_data(audio_file)
    # res = asyncio.run(asr.transcribe(session))
    # print(res)

    asr = WhisperCPPCstyleAsr(model_name_or_path="./models/ggml-base.bin")
    asr.set_audio_data(audio_file)
    res = asyncio.run(asr.transcribe(session))
    print(res)
