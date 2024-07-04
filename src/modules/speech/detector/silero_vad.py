import torch
import numpy as np

from src.common.session import Session
from src.common.interface import IDetector
from src.common.types import SileroVADArgs, RATE, INT16_MAX_ABS_VALUE
from src.common.factory import EngineClass


class SileroVAD(EngineClass, IDetector):
    TAG = "silero_vad"

    def __init__(self, **args: SileroVADArgs) -> None:
        self.args = SileroVADArgs(**args)
        self.model, _ = torch.hub.load(
            repo_or_dir=self.args.repo_or_dir,
            model=self.args.model,
            source=self.args.source,
            force_reload=self.args.force_reload,
            onnx=self.args.onnx,
            verbose=self.args.verbose,
        )

    async def detect(self, session: Session):
        audio_chunk = np.frombuffer(self.audio_buffer, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
        vad_prob = self.silero_vad_model(
            torch.from_numpy(audio_chunk),
            RATE).item()
        is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
        return is_silero_speech_active

    def get_sample_info(self):
        return RATE, int(len(self.audio_buffer) / 2)

    def set_audio_data(self, audio_data):
        self.audio_buffer = audio_data

    def close(self):
        pass
