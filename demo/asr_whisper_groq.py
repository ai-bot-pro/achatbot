import io
import os
from groq import Groq


def transcription(filename):
    with io.BytesIO() as f:
        with open(filename, "rb") as file:
            f.write(file.read())
        f.seek(0)

        transcription = client.audio.transcriptions.create(
            file=("oligei.wav", f),
            model="whisper-large-v3",
            prompt="",  # Optional
            response_format="verbose_json",  # Optional
            language="zh",  # Optional
            temperature=0.0,  # Optional
            timeout=None,
        )
        print(transcription)


def translations(filename):
    with open(filename, "rb") as file:
        translation = client.audio.translations.create(
            # file=(filename, file.read()),
            file=("oligei.wav", file.read()),
            # file=("oligei.wav", io.BytesIO(file.read())),
            model="whisper-large-v3",
            prompt="",  # Optional
            response_format="text",  # Optional
            temperature=0.0,  # Optional
        )
        print(translation)


if __name__ == "__main__":
    client = Groq()
    filename = "./records/tmp.wav"
    transcription(filename)
    translations(filename)
