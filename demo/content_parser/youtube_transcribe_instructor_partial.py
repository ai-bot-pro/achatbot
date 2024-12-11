import os

import instructor
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from rich.console import Console
from rich.table import Table
from rich.live import Live

from . import types
from .table import chapter

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(override=True)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",
    ),
    mode=instructor.Mode.GEMINI_JSON,
)


def get_youtube_transcript(video_id: str, languages=("en", "zh-CN")) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return " ".join([f"ts={entry['start']} - {entry['text']}" for entry in transcript])
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return ""


def extract_models(transcript: str, language="en"):
    class Chapters(BaseModel):
        chapters: list[chapter.Chapter]  # type: ignore

    return client.chat.completions.create_partial(
        response_model=Chapters,
        messages=[
            {
                "role": "system",
                "content": f"Analyze the given YouTube transcript and extract chapters. For each chapter, provide a start timestamp, end timestamp, title, and summary. Output language should be in {types.TO_LLM_LANGUAGE[language]}",
            },
            {"role": "user", "content": transcript},
        ],
    )


if __name__ == "__main__":
    video_id = input("Enter a Youtube Url: ")
    language = input("Enter Language(en/zh): ")
    video_id = video_id.split("v=")[1]
    console = Console()

    with console.status("[bold green]Processing YouTube URL...") as status:
        transcripts = get_youtube_transcript(video_id)
        status.update("[bold blue]Generating Clips...")
        chapters = extract_models(transcripts, language=language)

        table = Table(title="Video Chapters")
        table.add_column("Title", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Start", style="cyan")
        table.add_column("End", style="cyan")

        with Live(refresh_per_second=4) as live:
            for extraction in chapters:
                if not extraction.chapters:
                    continue

                new_table = Table(title="Video Chapters")
                new_table.add_column("Title", style="magenta")
                new_table.add_column("Description", style="green")
                new_table.add_column("Start", style="cyan")
                new_table.add_column("End", style="cyan")

                for chapter in extraction.chapters:
                    new_table.add_row(
                        chapter.title,
                        chapter.summary,
                        f"{chapter.start_ts:.2f}" if chapter.start_ts else "",
                        f"{chapter.end_ts:.2f}" if chapter.end_ts else "",
                    )
                    new_table.add_row("", "", "", "")  # Add an empty row for spacing

                live.update(new_table)

    console.print("\nChapter extraction complete!")
