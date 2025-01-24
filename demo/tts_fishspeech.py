## fish-speech
# - https://weedge.github.io/post/multimoding/voices/fishspeech/

import io
import os
from pathlib import Path
import sys
import logging
import time
from typing import Optional

import click
import typer
import torch
import numpy as np
import soundfile
import torchaudio
import hydra
from hydra import compose, initialize
from hydra.utils import instantiate

try:
    cur_dir = os.path.dirname(__file__)
    sys.path.insert(1, os.path.join(cur_dir, "../deps/FishSpeech"))
    from deps.FishSpeech.fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
    from deps.FishSpeech.fish_speech.models.text2semantic.inference import load_model, generate_long
    from deps.FishSpeech.fish_speech.utils.file import AUDIO_EXTENSIONS
except ModuleNotFoundError as e:
    logging.error(
        "In order to use fishspeech-tts, you need to `pip install achatbot[tts_fishspeech]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.types import MODELS_DIR, RECORDS_DIR


app = typer.Typer()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def print_model_params(model: torch.nn.Module):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.debug(model)
    logging.debug(f"{model_million_params} M parameters")


def load_gan_model(
    checkpoint_path: str = os.path.join(
        MODELS_DIR, "fishaudio/fish-speech-1.5", "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    ),
    config_name: str = "firefly_gan_vq",
    config_path: str = "../deps/FishSpeech/fish_speech/configs",
    device: str = "cuda",
):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(checkpoint_path, map_location=device, mmap=True, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v for k, v in state_dict.items() if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    logging.info(f"Loaded model: {result}")
    return model


@app.command("encode_codebook_indices")
def encode_codebook_indices(
    input_path: Path,
    output_path: Path,
    checkpoint_path: str = os.path.join(
        MODELS_DIR, "fishaudio/fish-speech-1.5", "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    ),
    config_path: str = "../deps/FishSpeech/fish_speech/configs",
    config_name: str = "firefly_gan_vq",
    device: str | None = None,
    is_save: bool = True,
) -> torch.Tensor:
    """
    generate codebook indices from audio file
    """

    # load firefly-gan-vq-fsq (ConvNeXt Encoder and Firefly Generator) model
    device = device or get_device()
    model: FireflyArchitecture = load_gan_model(
        checkpoint_path=checkpoint_path,
        config_name=config_name,
        config_path=config_path,
        device=device,
    )
    print_model_params(model)

    # load audio
    if input_path.suffix in AUDIO_EXTENSIONS:
        logging.info(f"Processing in-place reconstruction of {input_path}")

        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, model.spec_transform.sample_rate)

        audios = audio[None].to(device)
        logging.info(
            f"Loaded audio with {audios.shape[2] / model.spec_transform.sample_rate:.2f} seconds"
        )

        # Firefly-GAN Encoder
        # 1. LogMelSpectrogram with STFT (torch.stft) input audio waveform transform to Mel spec
        # 2. ConvNeXt Encoder input Mel spec encode(downsample) to tensor $z_d$
        # 3. DownsampleFiniteScalarQuantize with grouped FSQ input downsampled tensor $z_d$ encode((downsample) to vq codebook indices (quantized Mel spec)
        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        indices = model.encode(audios, audio_lengths)[0][0]

        logging.info(f"Generated indices of shape {indices.shape}")

        # Save indices (.npy store numpy array)
        if is_save is True:
            output_path = output_path.with_suffix(".npy")
            np.save(output_path, indices.cpu().numpy())
            logging.info(f"Save indices numpy array to {output_path}")

        return indices


# @torch.no_grad()
@app.command("gen_waveform")
def gen_waveform(
    codebook_indices_path: Path,
    waveform_output_path: Path,
    checkpoint_path: str = os.path.join(
        MODELS_DIR, "fishaudio/fish-speech-1.5", "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    ),
    config_path: str = "../deps/FishSpeech/fish_speech/configs",
    config_name: str = "firefly_gan_vq",
    device: str | None = None,
    is_save: bool = True,
) -> torch.Tensor:
    """
    generate waveform from codebook indices
    """
    if codebook_indices_path.suffix != ".npy":
        raise ValueError("input_path must be a .npy file which store numpy array")

    # load firefly-gan-vq-fsq (ConvNeXt Encoder and Firefly Generator with grouped FSQ) model
    device = device or get_device()
    model: FireflyArchitecture = load_gan_model(
        checkpoint_path=checkpoint_path,
        config_name=config_name,
        config_path=config_path,
        device=device,
    )
    print_model_params(model)

    # load indices (codebook_num, feature_num)
    logging.info(f"Processing precomputed indices from {codebook_indices_path}")
    indices = np.load(codebook_indices_path)
    indices = torch.from_numpy(indices).to(device).long()
    assert indices.ndim == 2, f"Expected 2D indices (codebook_num, feature_num), got {indices.ndim}"
    logging.debug(f"Loaded indices of shape(codebook_num, feature_num) {indices.shape}, {indices}")

    # Firefly-GAN Decoder
    # 1. DownsampleFiniteScalarQuantize with grouped FSQ input vq codebook indices decode (upsample) to Mel spec
    # 2. Firefly Generator (firefly.HiFiGANGenerator) input Mel spec to waveform
    feature_lengths = torch.tensor([indices.shape[1]], device=device)
    waveform_tensor, _ = model.decode(indices=indices[None], feature_lengths=feature_lengths)
    audio_time = waveform_tensor.shape[-1] / model.spec_transform.sample_rate

    logging.info(
        f"Generated audio of shape {waveform_tensor.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Save waveform audio
    if is_save is True:
        waveform_np = (
            waveform_tensor[0, 0].float().detach().cpu().numpy()
        )  # B=1 C=1, save waveform seq , the same as waveform_tensor[0][0], waveform_tensor.squeeze()
        soundfile.write(waveform_output_path, waveform_np, model.spec_transform.sample_rate)
        logging.info(f"Saved audio to {waveform_output_path}")


@app.command("gen_codebook_indices")
def gen_codebook_indices(
    lm_checkpoint_dir: str = str(os.path.join(MODELS_DIR, "fishaudio/fish-speech-1.5")),
    text: str = "weedge,🐂niubility!🍺",
    prompt_text: list[str] = [],
    prompt_tokens: list[Path] = [],
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    temperature: float = 0.7,
    device: str | None = None,
    compile: bool = False,
    seed: int = 42,
    half: bool = False,
    iterative_prompt: bool = True,
    chunk_length: int = 100,
    output_codebook_indices_dir: str = os.path.join(MODELS_DIR, "fishspeech_codebook_indices"),
) -> None:
    if prompt_text is not None and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )

    precision = torch.half if half else torch.bfloat16
    device = device or get_device()
    logging.info(f"device {device}")

    os.makedirs(output_codebook_indices_dir, exist_ok=True)

    logging.info(f"Loading Dual-AR LM model from {lm_checkpoint_dir} ...")
    t0 = time.time()
    model, decode_one_token = load_model(lm_checkpoint_dir, device, precision, compile=compile)
    print_model_params(model)
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logging.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    if prompt_tokens is not None:
        prompt_tokens = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logging.info(f"Sampled text: {response.text}, codes: {response.codes.shape}")
        elif response.action == "next":
            if codes:
                file_path = f"{output_codebook_indices_dir}/codes_{idx}.npy"
                np.save(
                    file_path,
                    torch.cat(codes, dim=1).cpu().numpy(),
                )
                logging.info(f"Saved codes to {file_path}")
            logging.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logging.error(f"Error: {response}")


r"""
python -m demo.tts_fishspeech encode_codebook_indices ./records/asr_example_zh.wav ./models/fishspeech_ref_code_indices.npy
python -m demo.tts_fishspeech gen_waveform ./models/fishspeech_ref_code_indices.npy ./records/asr_example_zh_fishspeech_gen.wav

python -m demo.tts_fishspeech gen_codebook_indices --num-samples 2
python -m demo.tts_fishspeech gen_waveform ./models/fishspeech_codebook_indices/codes_1.npy ./records/codes_1_fishspeech_gen.wav

python -m demo.tts_fishspeech gen_codebook_indices --num-samples 2 \
  --text "hello world,你叫什么名字，能讲一个故事吗？" \
  --prompt-text "" \
  --prompt-tokens "./models/fishspeech_ref_code_indices.npy" 
python -m demo.tts_fishspeech gen_waveform ./models/fishspeech_codebook_indices/codes_0.npy ./records/codes_0_fishspeech_gen.wav
python -m demo.tts_fishspeech gen_waveform ./models/fishspeech_codebook_indices/codes_1.npy ./records/codes_1_fishspeech_gen.wav

python -m demo.tts_fishspeech gen_codebook_indices --num-samples 2 \
  --text "hello world,你叫什么名字，能讲一个故事吗？" \
  --prompt-text "开心" \
  --prompt-tokens "./models/fishspeech_ref_code_indices.npy" 
python -m demo.tts_fishspeech gen_waveform ./models/fishspeech_codebook_indices/codes_0.npy ./records/codes_0_fishspeech_gen.wav
python -m demo.tts_fishspeech gen_waveform ./models/fishspeech_codebook_indices/codes_1.npy ./records/codes_1_fishspeech_gen.wav
"""

if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "info").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    app()
