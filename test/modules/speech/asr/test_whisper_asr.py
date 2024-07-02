r"""
ASR_TAG=whisper_asr python -m unittest test.modules.speech.asr.test_whisper_asr.TestWhisperASR.test_transcribe

ASR_TAG=whisper_timestamped_asr python -m unittest test.modules.speech.asr.test_whisper_asr.TestWhisperASR.test_transcribe

ASR_TAG=whisper_faster_asr python -m unittest test.modules.speech.asr.test_whisper_asr.TestWhisperASR.test_transcribe

ASR_TAG=whisper_transformers_asr MODEL_NAME_OR_PATH=./models/openai/whisper-base python -m unittest test.modules.speech.asr.test_whisper_asr.TestWhisperASR.test_transcribe

ASR_TAG=whisper_mlx_asr python -m unittest test.modules.speech.asr.test_whisper_asr.TestWhisperASR.test_transcribe
"""
import logging
import unittest
import os
import asyncio


from src.common.logger import Logger
from src.common.utils.audio_utils import save_audio_to_file, load_json, get_audio_segment
from src.common.factory import EngineFactory
from src.common.session import Session
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR
from src.modules.speech.asr.whisper_asr import EngineClass

r"""
KMP_DUPLICATE_LIB_OK=TRUE MODEL_NAME_OR_PATH=./models/Systran/faster-whisper-base ASR_VERBOSE=True ASR_TAG=whisper_faster_asr python -m unittest test.modules.speech.asr.test_whisper_asr.TestWhisperASR.test_transcribe
"""


class TestWhisperASR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        audio_file = os.path.join(
            RECORDS_DIR, f"tmp.wav")
        # Use an environment variable to get the ASR model TAG
        cls.asr_tag = os.getenv('ASR_TAG', "whisper_faster_asr")
        cls.audio_file = os.getenv('AUDIO_FILE', audio_file)
        cls.verbose = os.getenv('ASR_VERBOSE', "")
        cls.model_name_or_path = os.getenv('MODEL_NAME_OR_PATH', 'base')
        Logger.init(logging.DEBUG)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["model_name_or_path"] = self.model_name_or_path
        kwargs["download_path"] = MODELS_DIR
        kwargs["verbose"] = bool(self.verbose)
        self.asr = EngineFactory.get_engine_by_tag(
            EngineClass, self.asr_tag, **kwargs)

        self.annotations_path = os.path.join(
            TEST_DIR, "audio_files/annotations.json")

        self.session = Session(**SessionCtx(
            "test_client_id", 16000, 2).__dict__)

    def tearDown(self):
        pass

    def test_transcribe(self):
        self.asr.set_audio_data(self.audio_file)
        self.asr.args.language = "zh"
        res = asyncio.run(
            self.asr.transcribe(self.session))
        print(res)

    def test_transcribe_segments(self):
        from sentence_transformers import SentenceTransformer, util
        self.similarity_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2')
        annotations = asyncio.run(load_json(self.annotations_path))

        for audio_file, data in annotations.items():
            audio_file_path = os.path.join(
                TEST_DIR, f"audio_files/{audio_file}")

            similarities = []
            for segment in data["segments"]:
                audio_segment = asyncio.run(get_audio_segment(
                    audio_file_path,
                    segment["start"],
                    segment["end"]))
                audio_frames = bytearray(audio_segment.raw_data)
                self.asr.args.language = "en"

                file_path = asyncio.run(save_audio_to_file(
                    audio_frames, self.session.get_record_audio_name(),
                    audio_dir=RECORDS_DIR))
                self.asr.set_audio_data(file_path)

                transcription = asyncio.run(
                    self.asr.transcribe(self.session))["text"]

                os.remove(file_path)

                embedding_1 = self.similarity_model.encode(
                    transcription.lower().strip(), convert_to_tensor=True)
                embedding_2 = self.similarity_model.encode(
                    segment["transcription"].lower().strip(), convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(
                    embedding_1, embedding_2).item()
                similarities.append(similarity)

                print(
                    f"\nSegment from '{audio_file}' ({segment['start']}-{segment['end']}s):")
                print(f"Expected: {segment['transcription']}")
                print(f"Actual: {transcription}")
                # self.assertGreater(len(transcription), 0)

            # Calculate average similarity for the file
            avg_similarity = sum(similarities) / len(similarities)
            print(f"\nAverage similarity for '{audio_file}': {avg_similarity}")

            # Assert that the average similarity is above the threshold
            # Adjust the threshold as needed
            self.assertGreaterEqual(avg_similarity, 0.7)
