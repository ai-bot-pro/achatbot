from pydantic import ConfigDict
from pydantic.main import BaseModel


class TransportParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_out_enabled: bool = False
    camera_out_is_live: bool = False
    camera_out_width: int = 1024
    camera_out_height: int = 768
    camera_out_bitrate: int = 800000
    camera_out_framerate: int = 30
    camera_out_color_format: str = "RGB"
    audio_out_enabled: bool = False
    audio_out_sample_rate: int = 16000
    audio_out_channels: int = 1
    audio_in_enabled: bool = False
    audio_in_sample_rate: int = 16000
    audio_in_channels: int = 1
    vad_enabled: bool = False
    vad_audio_passthrough: bool = False

