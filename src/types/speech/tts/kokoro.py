from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR


@dataclass
class KokoroTTSArgs:
    device: str = None
    ckpt_path: str = os.path.join(MODELS_DIR, "Kokoro82M/kokoro-v0_19.pth")
    voices_stats_dir: str = os.path.join(MODELS_DIR, "Kokoro82M/voices")  # for load all voices
    voice: str = "af"  # Default voice is a 50-50 mix of Bella & Sarah

    # language is determined by the first letter of the VOICE_NAME:
    # 🇺🇸 'a' => American English => en-us
    # 🇬🇧 'b' => British English => en-gb
    # so kokoro-v0_19 just support English
    language: str = "a"

    # generate params
    speed: float = 1.0

    # stream
    tts_stream: bool = False
    chunk_length_seconds: int = 1

    # silence
    add_silence_chunk: bool = False


@dataclass
class KokoroOnnxTTSArgs:
    # device: str = None
    model_struct_stats_ckpt: str = os.path.join(MODELS_DIR, "Kokoro82M/kokoro-v0_19.onnx")
    voices_file_path: str = os.path.join(MODELS_DIR, "Kokoro82M/kokoro-voices.json")
    espeak_ng_data_path: str = None
    espeak_ng_lib_path: str = None  # "/usr/local/lib/libespeak-ng.1.dylib"
    voice: str = "af"  # Default voice is a 50-50 mix of Bella & Sarah

    # # https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
    # "en-us",  # English
    # "en-gb",  # English (British)
    # "fr-fr",  # French
    # "ja",  # Japanese
    # "ko",  # Korean
    # "cmn",  # Mandarin Chinese
    language: str = "en-us"

    # generate params
    speed: float = 1.0

    # stream
    tts_stream: bool = False
    chunk_length_seconds: int = 1

    # silence
    add_silence_chunk: bool = False
