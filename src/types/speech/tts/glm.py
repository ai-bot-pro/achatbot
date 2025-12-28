from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR


@dataclass
class GLMTTSTTSArgs:
    """
    llama + difusion(flow) + vocos(hift)  -> glm TTS
    """

    device: str | None = None
    model_path: str = os.path.join(MODELS_DIR, "zai-org/GLM-TTS")

    # prompt text and speech (one short for voice clone)
    default_voice_name: str = "jiayan_zh"
    default_prompt_text: str = "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"
    default_prompt_speech_path: str = "https://raw.githubusercontent.com/weedge/GLM-TTS/refs/heads/main/examples/prompt/jiayan_zh.wav"

    # generate params
    sample_method: str = "ras"
    use_phoneme: bool = False
    inference_steps: int = 10

    # random seed
    seed: int = 42

    # warm up
    warmup_steps: int = 1
    warm_up_text: str = "hello world"

    # streaming
    tts_stream: bool = True
    chunk_length_seconds: float = 0.5

    # silence
    add_silence_chunk: bool = False
