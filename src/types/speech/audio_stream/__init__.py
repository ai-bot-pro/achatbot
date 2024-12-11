from dataclasses import dataclass

from src.common.types import (
    CHANNELS,
    CHUNK,
    SAMPLE_WIDTH,
    RATE,
    PYAUDIO_PAINT16,
    AudioStreamArgs,
)

# pyaudio format
#   >>> print(int(pyaudio.paInt16))
#   8
#   >>> print(int(pyaudio.paInt24))
#   4
#   >>> print(int(pyaudio.paInt32))
#   2
#   >>> print(int(pyaudio.paFloat32))
#   1
#   >>> print(int(pyaudio.paInt8))
#   16
#   >>> print(int(pyaudio.paUInt8))
#   32
#   >>> print(int(pyaudio.paCustomFormat))
#   65536
FORMAT = PYAUDIO_PAINT16


@dataclass
class AudioStreamInfo:
    in_channels: int = CHANNELS
    in_sample_rate: int = RATE
    in_sample_width: int = SAMPLE_WIDTH
    in_frames_per_buffer: int = CHUNK
    out_channels: int = CHANNELS
    out_sample_rate: int = RATE
    out_sample_width: int = SAMPLE_WIDTH
    out_frames_per_buffer: int = CHUNK
    pyaudio_out_format: int | None = FORMAT


@dataclass
class PyAudioStreamArgs(AudioStreamArgs):
    format: int = FORMAT
    channels: int = CHANNELS
    rate: int = RATE
    sample_width: int = SAMPLE_WIDTH
    input_device_index: int = None
    output_device_index: int = None
