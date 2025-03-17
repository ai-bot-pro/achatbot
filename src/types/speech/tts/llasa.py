from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR
from src.types.codec import CodecArgs
from src.types.llm.transformers import TransformersSpeechLMArgs


@dataclass
class LlasaTTSArgs:
    """
    TransformersManualSpeechLlasa LM + xcodec2 -> llasa TTS
    """

    lm_args: dict = field(default_factory=lambda: TransformersSpeechLMArgs().__dict__)
    xcode2_args: dict = field(default_factory=lambda: CodecArgs().__dict__)

    # ref audio file path and prompt text
    ref_audio_file_path: str = ""
    prompt_text: str = ""
    # is save ref audio encode vq code indices .npy
    is_save: bool = False
    output_codebook_indices_dir: str = os.path.join(MODELS_DIR, "llasa_codebook_indices")
