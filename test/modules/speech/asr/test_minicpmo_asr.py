import logging
import unittest
import os
import asyncio


from src.common.logger import Logger
from src.common.utils.helper import load_json, get_audio_segment
from src.common.utils.wav import save_audio_to_file
from src.common.session import Session
from src.common.interface import IAsr
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR
from src.modules.speech.asr import ASREnvInit


r"""
LLM_MODEL_NAME_OR_PATH=./models/openbmb/MiniCPM-o-2_6 \
    LLM_GEN_TEMPERATURE=0.0 \
    LLM_DEVICE=cuda LLM_TORCH_DTYPE=bfloat16 \
    python -m unittest test.modules.speech.asr.test_minicpmo_asr.TestMiniCPMoASR.test_transcribe_stream

LLM_MODEL_NAME_OR_PATH=./models/openbmb/MiniCPM-o-2_6-int4 USE_GPTQ_CKPT=1 \
    LLM_GEN_TEMPERATURE=0.0 \
    LLM_DEVICE=cuda LLM_TORCH_DTYPE=bfloat16 \
    python -m unittest test.modules.speech.asr.test_minicpmo_asr.TestMiniCPMoASR.test_transcribe_stream

LLM_MODEL_NAME_OR_PATH=./models/openbmb/MiniCPM-o-2_6 \
    LLM_GEN_TEMPERATURE=0.0 \
    LLM_DEVICE=cuda LLM_TORCH_DTYPE=bfloat16 \
    python -m unittest test.modules.speech.asr.test_minicpmo_asr.TestMiniCPMoASR.test_transcribe

LLM_MODEL_NAME_OR_PATH=./models/openbmb/MiniCPM-o-2_6-int4 USE_GPTQ_CKPT=1 \
    LLM_GEN_TEMPERATURE=0.0 \
    LLM_DEVICE=cuda LLM_TORCH_DTYPE=bfloat16 \
    python -m unittest test.modules.speech.asr.test_minicpmo_asr.TestMiniCPMoASR.test_transcribe

LLM_MODEL_NAME_OR_PATH=./models/openbmb/MiniCPM-o-2_6 \
    LLM_GEN_TEMPERATURE=0.0 \
    LLM_DEVICE=cuda LLM_TORCH_DTYPE=bfloat16 \
    python -m unittest test.modules.speech.asr.test_minicpmo_asr.TestMiniCPMoASR.test_transcribe_with_bytes
"""


class TestMiniCPMoASR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # wget
        # https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
        # -O records/asr_example_zh.wav
        audio_file = os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav")
        # Use an environment variable to get the ASR model TAG
        cls.asr_tag = os.getenv("ASR_TAG", "minicpmo_asr")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)

        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.asr: IAsr = ASREnvInit.initASREngine(self.asr_tag)

        self.annotations_path = os.path.join(TEST_DIR, "audio_files/annotations.json")

        self.session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)

    def tearDown(self):
        pass

    def test_transcribe_stream(self):
        self.asr.set_audio_data(self.audio_file)
        res = self.asr.transcribe_stream_sync(self.session)
        for word in res:
            print(word)
            self.assertGreater(len(word), 0)

    def test_transcribe(self):
        self.asr.set_audio_data(self.audio_file)
        res = asyncio.run(self.asr.transcribe(self.session))
        print(res)

    def test_transcribe_with_bytes(self):
        with open(self.audio_file, "rb") as file:
            self.asr.set_audio_data(file.read())
            res = asyncio.run(self.asr.transcribe(self.session))
            print(res)

    def test_transcribe_with_record(self):
        import pyaudio

        paud = pyaudio.PyAudio()
        audio_stream = paud.open(
            rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=1024
        )

        audio_stream.start_stream()
        logging.debug("start recording")
        while True:
            # empty, need use vad
            read_audio_frames = audio_stream.read(512)
            self.asr.set_audio_data(read_audio_frames)
            res = asyncio.run(self.asr.transcribe(self.session))
            logging.info(res)
            if len(res) > 0:
                break

        audio_stream.stop_stream()
        audio_stream.close()
        paud.terminate()

    def test_transcribe_segments(self):
        from sentence_transformers import SentenceTransformer, util

        self.similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        annotations = asyncio.run(load_json(self.annotations_path))

        for audio_file, data in annotations.items():
            audio_file_path = os.path.join(TEST_DIR, f"audio_files/{audio_file}")

            similarities = []
            for segment in data["segments"]:
                audio_segment = asyncio.run(
                    get_audio_segment(audio_file_path, segment["start"], segment["end"])
                )
                audio_frames = bytearray(audio_segment.raw_data)

                file_path = asyncio.run(
                    save_audio_to_file(
                        audio_frames, self.session.get_record_audio_name(), audio_dir=RECORDS_DIR
                    )
                )
                self.asr.set_audio_data(file_path)
                # self.asr.set_audio_data(audio_frames)

                transcription = asyncio.run(self.asr.transcribe(self.session))["text"]

                os.remove(file_path)

                embedding_1 = self.similarity_model.encode(
                    transcription.lower().strip(), convert_to_tensor=True
                )
                embedding_2 = self.similarity_model.encode(
                    segment["transcription"].lower().strip(), convert_to_tensor=True
                )
                similarity = util.pytorch_cos_sim(embedding_1, embedding_2).item()
                similarities.append(similarity)

                print(f"\nSegment from '{audio_file}' ({segment['start']}-{segment['end']}s):")
                print(f"Expected: {segment['transcription']}")
                print(f"Actual: {transcription}")
                # self.assertGreater(len(transcription), 0)

            # Calculate average similarity for the file
            avg_similarity = sum(similarities) / len(similarities)
            print(f"\nAverage similarity for '{audio_file}': {avg_similarity}")

            # Assert that the average similarity is above the threshold
            # Adjust the threshold as needed
            self.assertGreaterEqual(avg_similarity, 0.7)
