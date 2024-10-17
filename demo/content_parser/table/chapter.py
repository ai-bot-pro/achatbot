from typing import Generator
from pydantic import BaseModel, Field


class Chapter(BaseModel):
    start_ts: float = Field(
        ...,
        description="The start timestamp indicating when the chapter starts in the video.",
    )
    end_ts: float = Field(
        ...,
        description="The end timestamp indicating when the chapter ends in the video.",
    )
    title: str = Field(
        ..., description="A concise and descriptive title for the chapter."
    )
    summary: str = Field(
        ...,
        description="A brief summary of the chapter's content, don't use words like 'the speaker'",
    )


class Chapters(BaseModel):
    chapters: list[Chapter]


def console_table(chapters: Generator[Chapter, None, None]):
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
