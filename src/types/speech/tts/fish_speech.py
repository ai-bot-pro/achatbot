import os
from pathlib import Path
from typing_extensions import Annotated
from pydantic import BaseModel, Field, conint

from src.common.types import MODELS_DIR


class FishSpeechTTSInferArgs(BaseModel):
    warm_up_text: str = "weedge,🐂niubility!🍺"
    ref_text: str | None = ""
    # ref_encode_codebook_indices file path
    ref_audio_path: str | None = None
    # ref prompt tokens cache, @todo: get/set from dist kv store cache/db
    # use_memory_cache: Literal["on", "off"] = "off"

    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # split text into smaller chunks
    # if preprocess is done, set False
    iterative_prompt: bool = True
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200

    # use mask self-attention kv cache
    kv_cache: bool = True

    # if seq len > max_length, gen break
    max_length: int = 2048

    num_samples: int = 1
    # sampling generate (prefill_decode(decode first token for TTFT(Time to First Token)) -> decode_n_tokens for TPOT(Time per Output Token)),
    # TODO(LMOps): LM inference serving dist deploy
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.2
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    # if prompt_tokens + max_new_tokens > max_seq_len, max_new_tokens = max_seq_len - prompt_tokens
    # if max_new_tokens==0, max_new_tokens = max_seq_len - prompt_tokens
    # max_seq_len:8192 from model config:
    # https://huggingface.co/fishaudio/fish-speech-1.5/blob/main/config.json
    max_new_tokens: int = 1024

    # random seed
    seed: int = 42

    # streaming
    tts_stream: bool = True
    chunk_length_seconds: float = 0.5

    # silence
    add_silence_chunk: bool = False


class FishSpeechTTSDualARLMArgs(FishSpeechTTSInferArgs):
    lm_checkpoint_dir: str = str(os.path.join(MODELS_DIR, "fishaudio/fish-speech-1.5"))
    half: bool = False
    device: str | None = None
    # torch compile to accelerate inference
    compile: bool = False

    # save gen result
    is_save: bool = False
    output_codebook_indices_dir: str = os.path.join(MODELS_DIR, "fishspeech_codebook_indices")


class FishSpeechTTSFireFlyGANArgs(BaseModel):
    # TODO(LMOps): GAN(Encoder + Generator) generate serving dist deploy
    gan_checkpoint_path: str = os.path.join(
        MODELS_DIR, "fishaudio/fish-speech-1.5", "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    )
    # need use relative path
    gan_config_path: str = "../../../../deps/FishSpeech/fish_speech/configs"
    gan_config_name: str = "firefly_gan_vq"
    device: str | None = None


class FishSpeechTTSFireFlyGANEncoderArgs(FishSpeechTTSFireFlyGANArgs):
    waveform_input_path: str | Path
    codebook_indices_output_path: str | Path
    # save gen result
    is_save: bool = False


class FishSpeechTTSFireFlyGANGeneratorArgs(FishSpeechTTSFireFlyGANArgs):
    codebook_indices_input_path: str | Path
    waveform_output_path: str | Path
    # save gen result
    is_save: bool = False


class FishSpeechTTSArgs(FishSpeechTTSDualARLMArgs, FishSpeechTTSFireFlyGANArgs):
    """
    Dual ARLM + GAN
    """
