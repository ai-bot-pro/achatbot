import os
import io
import json
import asyncio
import logging

import unittest
import numpy as np
import torch

from src.common.logger import Logger
from src.common.utils.audio_utils import save_audio_to_file, load_json, get_audio_segment
from src.common.factory import EngineFactory
from src.common.session import Session
from src.common.types import SessionCtx, TEST_DIR, MODELS_DIR, RECORDS_DIR, INT16_MAX_ABS_VALUE
from src.modules.speech.detector.pyannote import EngineClass


class TestPyannoteDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        audio_file = os.path.join(
            RECORDS_DIR, f"tmp.wav")
        cls.model_type = os.getenv(
            'MODEL_TYPE', 'segmentation-3.0')
        model_ckpt_path = os.path.join(
            MODELS_DIR, 'pyannote', cls.model_type, "pytorch_model.bin")
        cls.path_or_hf_repo = os.getenv(
            'PATH_OR_HF_REPO', model_ckpt_path)
        cls.tag = os.getenv('DETECTOR_TAG', "pyannote_vad")
        cls.audio_file = os.getenv('AUDIO_FILE', audio_file)

        Logger.init(logging.DEBUG)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        kwargs = {}
        kwargs["path_or_hf_repo"] = self.path_or_hf_repo
        kwargs["model_type"] = self.model_type
        self.detector = EngineFactory.get_engine_by_tag(
            EngineClass, self.tag, **kwargs)

        self.annotations_path = os.path.join(
            TEST_DIR, "audio_files/annotations.json")

        self.session = Session(**SessionCtx(
            "test_client_id", 16000, 2).__dict__)

    def tearDown(self):
        pass

    def test_detect(self):
        self.session.ctx.vad_pyannote_audio = self.audio_file
        logging.debug(self.session.ctx)
        res = asyncio.run(
            self.detector.detect(self.session))
        logging.debug(res)

    def test_detect_activity(self):
        annotations = asyncio.run(load_json(self.annotations_path))

        for audio_file, data in annotations.items():
            audio_file_path = os.path.join(
                TEST_DIR, f"audio_files/{audio_file}")

            for annotated_segment in data["segments"]:
                print('\n')
                # Load the specific audio segment for VAD
                audio_segment = asyncio.run(get_audio_segment(
                    audio_file_path,
                    annotated_segment["start"],
                    annotated_segment["end"]))

                self.assertEqual(audio_segment.frame_rate, 16000)

                frames = bytearray(audio_segment.raw_data)
                # Convert the buffer frames to a NumPy array
                audio_array = np.frombuffer(frames, dtype=np.int16)
                # Normalize the array to a [-1, 1] range
                float_data = audio_array.astype(
                    np.float32) / INT16_MAX_ABS_VALUE
                waveform_tensor = torch.tensor(float_data, dtype=torch.float32)
                # don't Stereo, just Mono, reshape(1,-1) (1(channel),size(time))
                if waveform_tensor.ndim == 1:
                    # float_data= float_data.reshape(1, -1)
                    waveform_tensor = waveform_tensor.reshape(1, -1)

                self.session.ctx.vad_pyannote_audio = {
                    "waveform": waveform_tensor, "sample_rate": 16000}

                # audio_frames = bytearray(audio_segment.raw_data)
                # file_path = asyncio.run(save_audio_to_file(
                #    audio_frames, self.session.get_file_name(),
                #    audio_dir=RECORDS_DIR))
                # self.session.ctx.vad_pyannote_audio = file_path

                vad_results = asyncio.run(
                    self.detector.detect(self.session))

                # os.remove(file_path)

                # Adjust VAD-detected times by adding the start time of the annotated segment
                adjusted_vad_results = [{"start": segment["start"] + annotated_segment["start"],
                                         "end": segment["end"] + annotated_segment["start"]}
                                        for segment in vad_results]

                detected_segments = [segment for segment in adjusted_vad_results if
                                     segment["start"] <= annotated_segment["start"] + 1.0 and
                                     segment["end"] <= annotated_segment["end"] + 2.0]

                # Print formatted information about the test
                print(
                    f"Testing segment from '{audio_file}': Annotated Start: {annotated_segment['start']}, Annotated End: {annotated_segment['end']}")
                print(f"VAD segments: {adjusted_vad_results}")
                print(f"Overlapping, Detected segments: {detected_segments}")

                # Assert that at least one detected segment meets the condition
                self.assertTrue(len(detected_segments) > 0,
                                "No detected segment matches the annotated segment")
