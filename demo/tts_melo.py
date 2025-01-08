## melo-tts
# - https://weedge.github.io/post/multimoding/voices/open_voice_extra_se_and_convert/
# - https://github.com/myshell-ai/MeloTTS

import os

import numpy as np
import soundfile
from melo.api import TTS

from src.common.types import MODELS_DIR, RECORDS_DIR

# Speed is adjustable
speed = 1.0
device = "cpu"  # or cuda:0
language = "ZH"  # ZH or EN_NEWEST


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


def save(data: np.ndarray, rate: int):
    output_path = os.path.join(RECORDS_DIR, f"melo_tts_{language}.wav")
    soundfile.write(output_path, data, rate)
    print(output_path)


if __name__ == "__main__":
    # download model
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
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {model_million_params} M {model}")
    print(f"config params:{model.hps}")
    speaker_ids = model.hps.data.spk2id
    print(speaker_ids)

    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        print(speaker_id, speaker_key)

        text = "我最近在学习deep learning，希望能够在未来的artificial intelligence领域有所建树。"
        np_audio_data = model.tts_to_file(text, speaker_id, None, speed=speed)

        if os.getenv("IS_SAVE", None):
            save(np_audio_data, model.hps.data.sampling_rate)

        speak(np_audio_data, model.hps.data.sampling_rate)
