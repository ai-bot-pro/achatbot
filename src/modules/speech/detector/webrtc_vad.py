import logging
import os

from pyannote.audio.core.io import AudioFile

from src.common.session import Session
from src.common.interface import IDetector
from src.common.types import PyannoteDetectorArgs, RATE, CHUNK
from src.common.factory import EngineClass


class PyannoteVAD(IDetector):
    TAG = "pyannote_vad"

    def __init__(self, **args: PyannoteDetector) -> None:
        super().__init__(**args)

        from pyannote.audio.pipelines import VoiceActivityDetection
        self.pipeline = VoiceActivityDetection(segmentation=self.model)
        self.pipeline.instantiate(self.hyper_parameters)

    async def detect(self, session: Session):
        vad_res = self.pipeline(self.args.vad_pyannote_audio)
        # `vad_res` is a pyannote.core.Annotation instance containing speech regions
        vad_segments = []
        for segment in vad_res.itersegments():
            logging.debug(f"vad_segment: {segment}")
            vad_segments.append(
                {"start": segment.start, "end": segment.end, "confidence": 1.0})
        logging.debug(f"vad_segments: {vad_segments}")
        return vad_segments
