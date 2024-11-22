import os
import wave
from PIL import Image

from apipeline.frames.data_frames import AudioRawFrame, ImageRawFrame

from src.common.types import ASSETS_DIR

# script_dir = os.path.dirname(__file__)


def load_images(image_files):
    images = {}
    for file in image_files:
        # Build the full path to the image file
        # full_path = os.path.join(script_dir, "../assets", file)
        full_path = os.path.join(ASSETS_DIR, file)
        # Get the filename without the extension to use as the dictionary key
        filename = os.path.splitext(os.path.basename(full_path))[0]
        # Open the image and convert it to bytes
        with Image.open(full_path) as img:
            images[filename] = ImageRawFrame(
                image=img.tobytes(),
                size=img.size,
                format=img.format,
                mode="RGB",
            )
    return images


def load_sounds(sound_files):
    sounds = {}

    for file in sound_files:
        # Build the full path to the sound file
        # full_path = os.path.join(script_dir, "../assets", file)
        full_path = os.path.join(ASSETS_DIR, file)
        # Get the filename without the extension to use as the dictionary key
        filename = os.path.splitext(os.path.basename(full_path))[0]
        # Open the sound and convert it to bytes
        with wave.open(full_path) as audio_file:
            sounds[filename] = AudioRawFrame(
                audio=audio_file.readframes(-1),
                sample_rate=audio_file.getframerate(),
                num_channels=audio_file.getnchannels(),
            )

    return sounds
