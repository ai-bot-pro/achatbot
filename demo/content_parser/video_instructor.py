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
def describe_video_url(url: str, language: str = "en"):
    pass


@app.command()
def describe_video_file(file: str, language: str = "en"):
    genai_file_obj = genai.upload_file(file)
    print("google genai file obj", genai_file_obj)

    content = [
        f"Describe what's happening in this video. Output language should be in {TO_LLM_LANGUAGE[language]}",
        # f"Try to elaborate but don't say your are analyzing an video focus on the description. Output language should be in {TO_LLM_LANGUAGE[language]}",
        genai_file_obj,
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


def extract_models(file: str, language: str = "en"):
    genai_file_obj = genai.upload_file(file)
    print("google genai file obj", genai_file_obj)
    content = [
        "Try to elaborate but don't say your are analyzing an video focus on the description.",
        genai_file_obj,
    ]

    resp = client.create_partial(
        response_model=chapter.Chapters,
        messages=[
            {
                "role": "system",
                "content": f"Analyze the description and extract chapters. For each chapter, provide a start timestamp, end timestamp, title, and summary. Output language should be in {TO_LLM_LANGUAGE[language]}",
            },
            {"role": "user", "content": content},
        ],
    )
    return resp


@app.command()
def instruct_content(files: List[str], language: str = "en") -> None:
    console = Console()
    for file in files:
        try:
            with console.status(f"[bold green]Processing {file}...") as status:
                status.update("[bold blue]Generating chapters...")
                chapters = extract_models(file, language=language)
                chapter.console_table(chapters)

            console.print(f"\n{file} Chapter extraction complete!")
        except Exception as e:
            logging.error(f"An error occurred while processing {file}: {str(e)}")


r"""
# https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.http.MediaFileUpload-class.html#resumable

python -m demo.content_parser.video_instructor describe-video-file \
    "/Users/wuyong/Desktop/videoplayback.mp4" \
    --language zh
python -m demo.content_parser.video_instructor instruct-content \
    "/Users/wuyong/Desktop/videoplayback.mp4" \
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
