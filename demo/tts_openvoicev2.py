## OpenVoiceV2 & melo-tts
# - https://weedge.github.io/post/multimoding/voices/open_voice_extra_se_and_convert/
# - https://github.com/myshell-ai/MeloTTS

import io
import os
import sys
import logging

import torch
import numpy as np
import soundfile
from melo.api import TTS

try:
    cur_dir = os.path.dirname(__file__)
    sys.path.insert(1, os.path.join(cur_dir, "../deps/OpenVoice"))
    from deps.OpenVoice.openvoice import se_extractor
    from deps.OpenVoice.openvoice.api import ToneColorConverter
except ModuleNotFoundError as e:
    logging.error(
        "In order to use openvoice-tts, you need to `pip install achatbot[tts_openvoicev2]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.types import ASSETS_DIR, MODELS_DIR, RECORDS_DIR

# Speed is adjustable
speed = 1.0
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
language = "ZH"  # or EN_NEWEST


def load_tone_color_converter():
    # download openvoice converter model ckpt
    # huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir ./models/myshell-ai/OpenVoiceV2

    converter_conf_path = os.path.join(MODELS_DIR, "myshell-ai/OpenVoiceV2/converter/config.json")
    tone_color_converter = ToneColorConverter(converter_conf_path, device=device)

    # model_million_params = sum(p.numel() for p in tone_color_converter.model.parameters()) / 1e6
    # print(f"coverter model: {tone_color_converter.model}")
    # print(f"coverter model parameters: {model_million_params}M ")

    # model_million_params = (
    #    sum(p.numel() for p in tone_color_converter.watermark_model.parameters()) / 1e6
    # )
    # print(f"coverter watermark model: {tone_color_converter.watermark_model}")
    # print(f"coverter watermark model parameters: {model_million_params}M ")

    converter_ckpt_path = os.path.join(
        MODELS_DIR, "myshell-ai/OpenVoiceV2/converter/checkpoint.pth"
    )
    tone_color_converter.load_ckpt(converter_ckpt_path)

    return tone_color_converter


def reference_target_se_extractor(
    tone_color_converter: ToneColorConverter | None = None,
    reference_speaker_file: str = os.path.join(ASSETS_DIR, "basic_ref_zh.wav"),
):
    if tone_color_converter is None:
        tone_color_converter = load_tone_color_converter()
    # This is the voice you want to clone
    target_dir = os.path.join(RECORDS_DIR, "openvoicev2")
    os.makedirs(target_dir, exist_ok=True)
    vad = True  # False use whisper, True use silero vad
    target_se, audio_name = se_extractor.get_se(
        reference_speaker_file, tone_color_converter, target_dir=target_dir, vad=vad
    )
    se_path = os.path.join(target_dir, audio_name, "se.pth")
    print(f"saved target tone color file: {se_path}, target tone color shape: {target_se.shape}")

    return target_se, se_path


def speak(np_audio_data: np.ndarray, rate: int):
    import pyaudio

    # data = np.frombuffer(np_audio_data, dtype=np.float32).tobytes()
    data = np_audio_data.astype(dtype=np.float32).tobytes()

    pyaudio_instance = pyaudio.PyAudio()
    audio_stream = pyaudio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=rate,
        output_device_index=None,
        output=True,
    )

    audio_stream.write(data)

    audio_stream.stop_stream()
    audio_stream.close()
    pyaudio_instance.terminate()


def save(data: np.ndarray, rate: int, file_name: str = "melo_tts"):
    os.makedirs(RECORDS_DIR, exist_ok=True)
    output_path = os.path.join(RECORDS_DIR, f"{file_name}_{language}.wav")
    soundfile.write(output_path, data, rate)
    print(output_path)
    return output_path


def openvoice_clone(
    tone_color_converter: ToneColorConverter,
    target_se: torch.Tensor,
    target_se_path: str = "",
    watermark="@Weedge",
):
    if target_se_path:
        print(f"target_se load form {target_se_path}")
        target_se = torch.load(target_se_path, map_location=device)
    # download melo-tts model ckpt
    # huggingface-cli download myshell-ai/MeloTTS-English-v3 --local-dir ./models/myshell-ai/MeloTTS-English-v3
    # huggingface-cli download myshell-ai/MeloTTS-Chinese --local-dir ./models/myshell-ai/MeloTTS-Chinese

    config_path = os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-Chinese/config.json")
    ckpt_path = os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-Chinese/checkpoint.pth")
    if language == "EN_NEWEST":
        config_path = os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-English-v3/config.json")
        ckpt_path = os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-English-v3/checkpoint.pth")

    model = TTS(
        language=language,
        device=device,
        config_path=config_path,
        ckpt_path=ckpt_path,
    )
    # print the number of parameters in the model
    # model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    # print(f"melo tts model params: {model_million_params} M {model}")

    print(f"melo tts config params:{model.hps}")
    speaker_ids = model.hps.data.spk2id
    print(speaker_ids)

    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace("_", "-")
        print(speaker_id, speaker_key)

        text = "我们介绍了一种深度且轻量级的Transformer，名为DeLighT，它在参数数量显著减少的情况下，提供了与标准基于Transformer的模型相似或更好的性能。"
        np_audio_data = model.tts_to_file(text, speaker_id, None, speed=speed)

        src_path = ""
        if os.getenv("IS_SAVE", None):
            src_path = save(np_audio_data, model.hps.data.sampling_rate)

        speak(np_audio_data, model.hps.data.sampling_rate)

        if not src_path:
            print("no src_path, use soundfile")
            audio_buf = io.BytesIO()
            soundfile.write(
                audio_buf,
                np_audio_data.astype(dtype=np.float32),
                model.hps.data.sampling_rate,
                format="WAV",
            )
            audio_buf.seek(0)
            src_path = soundfile.SoundFile(audio_buf)

        # Run the tone color converter
        se_ckpt_path = os.path.join(
            MODELS_DIR, f"myshell-ai/OpenVoiceV2/base_speakers/ses/{speaker_key}.pth"
        )
        source_se = torch.load(se_ckpt_path, map_location=device)

        np_convert_audio_data = tone_color_converter.convert(
            audio_src_path=src_path,  # 使用 SoundFile 对象
            src_se=source_se,
            tgt_se=target_se,
            output_path=None,
            message=watermark,
        )
        # print(np_convert_audio_data)

        if isinstance(src_path, soundfile.SoundFile):
            # 关闭文件
            src_path.close()
            audio_buf.close()

        if os.getenv("IS_SAVE", None):
            save(
                np_convert_audio_data,
                tone_color_converter.hps.data.sampling_rate,
                file_name="openvoicev2_tts",
            )

        speak(np_convert_audio_data, tone_color_converter.hps.data.sampling_rate)


if __name__ == "__main__":
    tone_color_converter = load_tone_color_converter()
    target_se, se_path = reference_target_se_extractor(tone_color_converter)
    openvoice_clone(tone_color_converter, target_se, target_se_path=se_path)
