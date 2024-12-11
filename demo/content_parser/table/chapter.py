import os
from typing import Generator, List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai
import instructor

from .. import types

# Load environment variables from .env file
load_dotenv(override=True)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",
    ),
    mode=instructor.Mode.GEMINI_JSON,
    generation_config={
        # "max_output_tokens": 1024,
        # "temperature": 1.0,
        # "top_p": 0.1,
        # "top_k": 40,
        # "response_mime_type": "text/plain",
    },
)


def extract_models(content: str, mode="text", **kwargs):
    match mode:
        case "partial":
            return extract_models_partial(content, **kwargs)
        case "iterable":
            return extract_models_iterable(content, **kwargs)
        case _:
            return extract_models_text(content, **kwargs)


def extract_models_partial(content: str, **kwargs):
    sys_prompt = get_system_prompt(**kwargs)
    res = client.create_partial(
        response_model=Chapters,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {"role": "user", "content": content},
        ],
    )
    return res


def extract_models_iterable(content: str, **kwargs):
    sys_prompt = get_system_prompt(**kwargs)
    res = client.create_iterable(
        response_model=Chapters,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {"role": "user", "content": content},
        ],
    )
    return res


def extract_models_text(content: str, **kwargs):
    sys_prompt = get_system_prompt(**kwargs)
    res = client.create(
        response_model=Chapters,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {"role": "user", "content": content},
        ],
    )
    return res


class ChapterSystemPromptArgs(BaseModel):
    language: str = "en"


def get_system_prompt(**kwargs) -> str:
    r"""
    !NOTE: the same as ell use python function  :)
    """
    args = ChapterSystemPromptArgs(**kwargs)
    return f"Analyze the given YouTube transcript and extract chapters. For each chapter, provide a start timestamp, end timestamp, title, and summary. Output language should be in {types.TO_LLM_LANGUAGE[args.language]}"


class Chapter(BaseModel):
    start_ts: float = Field(
        ...,
        description="The start timestamp indicating when the chapter starts in the video.",
    )
    end_ts: float = Field(
        ...,
        description="The end timestamp indicating when the chapter ends in the video.",
    )
    title: str = Field(..., description="A concise and descriptive title for the chapter.")
    summary: str = Field(
        ...,
        description="A brief summary of the chapter's content, don't use words like 'the speaker'",
    )


class Chapters(BaseModel):
    chapters: list[Chapter]


def console_table(chapters: Generator[Chapters, None, None] | List[Chapter]):
    from rich.table import Table
    from rich.live import Live

    table = Table(title="Chapters")
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
