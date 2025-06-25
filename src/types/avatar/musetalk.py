import os
from pydantic import BaseModel, Field

from src.common.types import MODELS_DIR


class AvatarMuseTalkConfig(BaseModel):
    """Configuration class for MuseTalk avatar processor."""

    fps: int = 25  # Video frames per second
    batch_size: int = 5  # Batch size for processing audio and video frames
    avatar_video_path: str = ""  # Path to the initialization video
    avatar_model_dir: str = os.path.join(MODELS_DIR, "musetalk/avatar_model")
    force_create_avatar: bool = False  # Whether to force data regeneration
    debug: bool = False  # Enable debug mode
    debug_save_handler_audio: bool = False  # Enable debug mode
    debug_replay_speech_id: str = ""  # Enable debug mode
    input_audio_sample_rate: int = 16000  # input audio sample rate
    input_audio_slice_duration: int = 1  # 1 second duration slice
    # Internal algorithm sample rate, fixed at 16000, used for input audio resampling
    # WhisperFeatureExtractor was trained using a sampling rate of 16000.
    algo_audio_sample_rate: int = 16000
    output_audio_sample_rate: int = 16000  # Output audio sample rate (for resampling)
    model_dir: str = os.path.join(MODELS_DIR, "musetalk")  # Root directory for models
