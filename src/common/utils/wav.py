import os

import wave

from src.common.types import RECORDS_DIR


async def save_audio_to_file(
    audio_data, file_name, audio_dir=RECORDS_DIR, channles=1, sample_width=2, sample_rate=16000
):
    os.makedirs(audio_dir, exist_ok=True)

    file_path = os.path.join(audio_dir, file_name)

    with wave.open(file_path, "wb") as wav_file:
        wav_file.setnchannels(channles)  # Assuming mono audio
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    return file_path


async def read_audio_file(file_path):
    with wave.open(file_path, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return frames
