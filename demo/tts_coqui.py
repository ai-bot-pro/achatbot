#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸
https://github.com/coqui-ai/tts
https://docs.coqui.ai/en/latest/models/xtts.html

- model ckpt,config: https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json
- eg:
tts --model_path models/coqui/XTTS-v2 --config_path models/coqui/XTTS-v2/config.json --list_language_idx
tts --model_path models/coqui/XTTS-v2 --config_path models/coqui/XTTS-v2/config.json --list_speaker_idx
tts --model_path models/coqui/XTTS-v2 --config_path models/coqui/XTTS-v2/config.json \
    --text "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent." \
    --speaker_idx "Ana Florence" \
    --out_path records/tts_coqui_en.wav \
    --language_idx en
tts --model_path models/coqui/XTTS-v2 --config_path models/coqui/XTTS-v2/config.json \
    --text "我花了很长时间才形成一种声音，现在我有了它，我不会保持沉默。" \
    --speaker_idx "Claribel Dervla" \
    --out_path records/tts_coqui_zh.wav \
    --language_idx zh
tts --model_path models/coqui/XTTS-v2 --config_path models/coqui/XTTS-v2/config.json \
    --text "Bugün okula gitmek istemiyorum." \
    --speaker_wav records/tmp.wav \
    --out_path records/tts_coqui_clone_tr.wav \
    --language_idx tr
tts --model_path models/coqui/XTTS-v2 --config_path models/coqui/XTTS-v2/config.json \
    --text "Bugün okula gitmek istemiyorum." \
    --speaker_wav records/tmp.wav records/tmp_webrtcvad.wav \
    --out_path records/tts_coqui_multi_clone_tr.wav \
    --language_idx tr
tts --model_path models/coqui/XTTS-v2 --config_path models/coqui/XTTS-v2/config.json \
    --text "六一儿童节快乐." \
    --speaker_wav records/tmp.wav \
    --out_path records/tts_coqui_clone_zh.wav \
    --language_idx zh
tts --model_path models/coqui/XTTS-v2 --config_path models/coqui/XTTS-v2/config.json \
    --text "六一儿童节快乐." \
    --speaker_wav records/tmp.wav records/tmp_webrtcvad.wav \
    --out_path records/tts_coqui_multi_clone_zh.wav \
    --language_idx zh

🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸🐸
"""

import os
import time
import torch
import torchaudio

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from device_cuda import CUDAInfo

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


def list_models():
    # List available 🐸TTS models
    models = TTS.list_models()
    print(models)


def xtts_speaker(model_name, model_path, conf_file, reference_audio_path="records/tmp.wav"):
    # Init TTS
    tts = TTS(model_name=model_name, model_path=model_path, config_path=conf_file).to(device)

    # Run TTS
    tts.tts_to_file(
        text="Hello world!",
        language="en",
        speaker="Ana Florence",
        file_path="records/tts_coqui_en.wav",
    )
    tts.tts_to_file(
        text="你好!", language="zh", speaker="Claribel Dervla", file_path="records/tts_coqui_zh.wav"
    )
    # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    # wav = tts.tts(text="Hello world!", speaker_wav="records/my_voice.wav", language="en")
    # Text to speech to a file
    tts.tts_to_file(
        text="Hello world!",
        speaker_wav=reference_audio_path,
        language="en",
        file_path="records/tts_coqui_clone_en.wav",
    )
    tts.tts_to_file(
        text="你好!",
        speaker_wav=reference_audio_path,
        split_sentences=False,
        language="zh",
        file_path="records/tts_coqui_clone_zh.wav",
    )


def inference(model_path, conf_file, reference_audio_path="records/tmp.wav"):
    info = CUDAInfo()

    print("inference Loading model...")
    config = XttsConfig()
    config.load_json(conf_file)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=info.is_cuda)
    print(model)
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{model_million_params}M parameters")

    if info.is_cuda:
        model.cuda()

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[reference_audio_path]
    )

    print("Inference...")
    out = model.inference(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,  # Add custom parameters here
    )
    torchaudio.save(
        "records/tts_coqui_infer_clone_en.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000
    )

    out = model.inference(
        "我花了很长时间才形成自己的声音，现在我有了声音，我不会保持沉默。",
        "zh",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,  # Add custom parameters here
    )
    torchaudio.save(
        "records/tts_coqui_infer_clone_zh.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000
    )


def inference_streaming(model_path, conf_file, reference_audio_path="records/tmp.wav"):
    info = CUDAInfo()

    print("inference_streaming Loading model...")
    config = XttsConfig()
    config.load_json(conf_file)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=info.is_cuda)
    print(model)
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{model_million_params}M parameters")

    if info.is_cuda:
        model.cuda()

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[reference_audio_path]
    )

    print("Inference...")
    t0 = time.time()
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )

    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunck: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}", chunk.shape)
        wav_chuncks.append(chunk)
    wav = torch.cat(wav_chuncks, dim=0)
    torchaudio.save(
        "records/tts_coqui_infer_stream_clone_en.wav", wav.squeeze().unsqueeze(0).cpu(), 24000
    )

    t0 = time.time()
    chunks = model.inference_stream(
        "我花了很长时间才形成自己的声音，现在我有了声音，我不会保持沉默。",
        "zh",
        gpt_cond_latent,
        speaker_embedding,
    )

    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunck: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}", chunk.shape)
        wav_chuncks.append(chunk)
    wav = torch.cat(wav_chuncks, dim=0)
    torchaudio.save(
        "records/tts_coqui_infer_stream_clone_zh.wav", wav.squeeze().unsqueeze(0).cpu(), 24000
    )


if __name__ == "__main__":
    """
    python demo/tts_coqui.py -m models/coqui/XTTS-v2 -c models/coqui/XTTS-v2/config.json
    python demo/tts_coqui.py -o inference -m models/coqui/XTTS-v2 -c models/coqui/XTTS-v2/config.json
    python demo/tts_coqui.py -o inference_streaming -m models/coqui/XTTS-v2 -c models/coqui/XTTS-v2/config.json
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", "-o", type=str, default="xtts_speaker", help="op method")
    parser.add_argument(
        "--model_name", "-n", type=str, default="", help="firstly choose model_name"
    )
    parser.add_argument(
        "--model_path", "-m", type=str, default="models/coqui/XTTS-v2", help="model path"
    )
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="models/coqui/XTTS-v2/config.json",
        help="config file",
    )
    args = parser.parse_args()
    if args.op == "inference":
        inference(args.model_path, args.config_file)
    elif args.op == "inference_streaming":
        inference_streaming(args.model_path, args.config_file)
    else:
        xtts_speaker(args.model_name, args.model_path, args.config_file)
