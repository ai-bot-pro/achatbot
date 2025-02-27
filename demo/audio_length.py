import logging
import os
import requests
from io import BytesIO
import mutagen
from mutagen.mp3 import MP3
from mutagen.oggvorbis import OggVorbis
from mutagen.flac import FLAC
from mutagen.wave import WAVE
from mutagen.aiff import AIFF

import typer

app = typer.Typer()


def get_audio_file_length(audio_file_path: str) -> float | None:
    """
    Retrieves the length of an audio file path

    Args:
        audio_file: the audio file path.

    Returns:
        The length of the audio file in seconds, or None if the length could not be determined.
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: File not found: {audio_file_path}")
        return None
    file_size = os.path.getsize(audio_file_path)
    logging.info(f"os.path.getsize: {file_size}")
    return file_size


def get_audio_url_length(audio_url: str) -> float | None:
    """
    Retrieves the length of an audio file from a URL.

    Args:
        audio_url: The URL of the audio file.

    Returns:
        The length of the audio file in seconds, or None if the length could not be determined.
    """
    try:
        # headers = {"Range": "bytes=0-102400"}  # Get the first 100KB, should be enough for metadata.
        response = requests.get(audio_url, stream=False, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes

        content_length = response.headers.get("Content-Length")
        if content_length:
            file_size = int(content_length)
            logging.info(f"File size: {file_size} bytes")
            return file_size

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not retrieve audio file from URL: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


@app.command("get_audio_length")
def get_audio_length(audio: str) -> float | None:
    if audio.startswith("http"):
        res = get_audio_url_length(audio)
    else:
        res = get_audio_file_length(audio)

    return res


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[
            # logging.FileHandler("content_parser_tts.log"),
            logging.StreamHandler()
        ],
    )
    app()
