from dataclasses import dataclass

from src.common.types import RATE


@dataclass
class ASRArgs:
    download_path: str = ""
    model_name_or_path: str = "base"
    # asr
    # NOTE:
    # - openai-whisper or whispertimestamped use str(file_path)/np.ndarray/torch tensor
    # - transformers whisper use torch tensor/tf tensor
    # - faster whisper don't use torch tensor, use np.ndarray or str(file_path)/~BinaryIO~
    # - mlx whisper don't use torch tensor, use str(file_path)/np.ndarray/~mlx.array~
    # - funasr whisper, SenseVoiceSmall use str(file_path)/torch tensor
    # asr_audio: str | bytes | IO[bytes] | np.ndarray | torch.Tensor = None

    # https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
    language: str = "zh"
    verbose: bool = True
    prompt: str = ""
    sample_rate: int = RATE
    device: str | dict | None = None
