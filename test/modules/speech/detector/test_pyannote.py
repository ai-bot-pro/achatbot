import os
import io
import json
import asyncio
import logging

import unittest

from src.common.logger import Logger
from src.common.utils.helper import load_json, get_audio_segment
from src.common.utils.audio_utils import bytes2TorchTensorWith16
from src.common.factory import EngineFactory
from src.common.session import Session
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR
from src.common.interface import IDetector
from src.modules.speech.detector.pyannote import EngineClass

r"""
python -m unittest test.modules.speech.detector.test_pyannote.TestPyannoteDetector.test_detect_activity
"""


class TestPyannoteDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        audio_file = os.path.join(RECORDS_DIR, "tmp.wav")
        cls.model_type = os.getenv("MODEL_TYPE", "segmentation-3.0")
        model_ckpt_path = os.path.join(MODELS_DIR, "pyannote", cls.model_type, "pytorch_model.bin")
        cls.path_or_hf_repo = os.getenv("PATH_OR_HF_REPO", model_ckpt_path)
        cls.tag = os.getenv("DETECTOR_TAG", "pyannote_vad")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)

        Logger.init(os.getenv("LOG_LEVEL", "debug").upper())

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["path_or_hf_repo"] = self.path_or_hf_repo
        kwargs["model_type"] = self.model_type
        self.detector: IDetector = EngineFactory.get_engine_by_tag(EngineClass, self.tag, **kwargs)

        self.annotations_path = os.path.join(TEST_DIR, "audio_files/annotations.json")

        self.session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)

    def tearDown(self):
        pass

    def test_detect(self):
        self.detector.set_audio_data(self.audio_file)
        res = asyncio.run(self.detector.detect(self.session))
        logging.debug(res)

    def test_detect_activity(self):
        annotations = asyncio.run(load_json(self.annotations_path))

        for audio_file, data in annotations.items():
            audio_file_path = os.path.join(TEST_DIR, f"audio_files/{audio_file}")

            for annotated_segment in data["segments"]:
                print("\n")
                # Load the specific audio segment for VAD
                audio_segment = asyncio.run(
                    get_audio_segment(
                        audio_file_path, annotated_segment["start"], annotated_segment["end"]
                    )
                )

                self.assertEqual(audio_segment.frame_rate, 16000)

                frames = bytearray(audio_segment.raw_data)
                waveform_tensor = bytes2TorchTensorWith16(frames)
                vad_pyannote_audio = {"waveform": waveform_tensor, "sample_rate": 16000}
                self.detector.set_audio_data(vad_pyannote_audio)

                # audio_frames = bytearray(audio_segment.raw_data)
                # file_path = asyncio.run(save_audio_to_file(
                #    audio_frames, self.session.get_file_name(),
                #    audio_dir=RECORDS_DIR))
                # self.detector.set_audio_data(file_path)

                vad_res = asyncio.run(self.detector.detect(self.session))

                # os.remove(file_path)

                # Adjust VAD-detected times by adding the start time of the annotated segment
                adjusted_vad_res = [
                    {
                        "start": segment["start"] + annotated_segment["start"],
                        "end": segment["end"] + annotated_segment["start"],
                    }
                    for segment in vad_res
                ]

                detected_segments = [
                    segment
                    for segment in adjusted_vad_res
                    if segment["start"] <= annotated_segment["start"] + 1.0
                    and segment["end"] <= annotated_segment["end"] + 2.0
                ]

                # Print formatted information about the test
                print(
                    f"Testing segment from '{audio_file}': Annotated Start: {annotated_segment['start']}, Annotated End: {annotated_segment['end']}"
                )
                print(f"VAD segments: {adjusted_vad_res}")
                print(f"Overlapping, Detected segments: {detected_segments}")

                # Assert that at least one detected segment meets the condition
                self.assertTrue(
                    len(detected_segments) > 0, "No detected segment matches the annotated segment"
                )
