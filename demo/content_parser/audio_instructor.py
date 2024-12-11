import logging
import os
from typing import List

import typer
import instructor
from pydantic import BaseModel
import google.generativeai as genai
from rich.console import Console
from dotenv import load_dotenv

from .table import chapter
from .types import TO_LLM_LANGUAGE

# Load environment variables from .env file
load_dotenv(override=True)

app = typer.Typer()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",
    ),
    mode=instructor.Mode.GEMINI_JSON,
)

app = typer.Typer()


class Description(BaseModel):
    description: str


@app.command()
def describe_audio(audio_file: str, language: str = "en"):
    audio_file = genai.upload_file(audio_file)
    print("google genai audio_file", audio_file)

    content = [
        f"Summarize what's happening in this audio file and who the main speaker is. Output language should be in {TO_LLM_LANGUAGE[language]}",
        audio_file,
    ]

    resp = client.create(
        response_model=Description,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    print(resp)


def extract_chapters(audio_file: str, language: str = "en"):
    audio_file = genai.upload_file(audio_file)
    print("google genai audio_file", audio_file)
    content = [
        audio_file,
    ]

    resp = client.create_partial(
        response_model=chapter.Chapters,
        messages=[
            {
                "role": "system",
                "content": f"Analyze the given audio and extract chapters. For each chapter, provide a start timestamp, end timestamp, title, and summary. Output language should be in {TO_LLM_LANGUAGE[language]}",
            },
            {"role": "user", "content": content},
        ],
    )
    return resp


@app.command()
def instruct_content(audio_files: List[str], language: str = "en") -> None:
    console = Console()
    for audio_file in audio_files:
        try:
            with console.status("[bold green]Processing audio...") as status:
                status.update("[bold blue]Generating chapters...")
                chapters = extract_chapters(audio_file, language=language)
                chapter.console_table(chapters)

            console.print(f"\n{audio_file} Chapter extraction complete!")
        except Exception as e:
            logging.error(f"An error occurred while processing {audio_file}: {str(e)}")


r"""
# https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.http.MediaFileUpload-class.html#resumable
# https://en.wikipedia.org/wiki/File:HTTP_cookie.ogg

python -m demo.content_parser.audio_instructor describe-audio \
    /Users/wuyong/Desktop/HTTP_cookie.ogg \
    --language zh

python -m demo.content_parser.audio_instructor instruct-content \
    /Users/wuyong/Desktop/HTTP_cookie.ogg \
    --language zh
"""
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[
            # logging.FileHandler("extractor.log"),
            logging.StreamHandler()
        ],
    )
    app()
