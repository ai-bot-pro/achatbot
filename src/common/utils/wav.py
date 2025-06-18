import logging
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


def read_wav_to_bytes(file_path) -> tuple[bytes, int]:
    """
    - params: file_path
    - return bytes and smaple rate
    """
    try:
        with wave.open(file_path, "rb") as wav_file:
            params = wav_file.getparams()
            logging.info(
                f"Channels: {params.nchannels}, Sample Width: {params.sampwidth}, Frame Rate: {params.framerate}, Number of Frames: {params.nframes}"
            )

            frames = wav_file.readframes(params.nframes)
            return frames, params.framerate
    except wave.Error as e:
        logging.exception(f"Error reading WAV file: {e}")
        return None, None
