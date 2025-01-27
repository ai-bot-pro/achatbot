from dataclasses import dataclass
import os

from common.types import MODELS_DIR
from src.types.codec import CodecArgs
from src.types.llm.transformers import TransformersSpeechLMArgs


@dataclass
class LlasaTTSArgs:
    """
    TransformersManualSpeechLlasa LM + xcodec2 -> llasa TTS
    """

    lm_args: TransformersSpeechLMArgs = TransformersSpeechLMArgs()
    xcode2_args: CodecArgs = CodecArgs()

    # ref audio file path and prompt text
    ref_audio_file_path: str = ""
    prompt_text: str = ""
    # is save ref audio encode vq code indices .npy
    is_save: bool = False
    output_codebook_indices_dir: str = os.path.join(MODELS_DIR, "llasa_codebook_indices")
