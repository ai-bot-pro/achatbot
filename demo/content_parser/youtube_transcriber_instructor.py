import os
from typing import List
import logging

import typer
from rich.console import Console
from youtube_transcript_api import YouTubeTranscriptApi

from .table import table

app = typer.Typer()


class YouTubeTranscriber:
    def extract_transcript(self, video_id: str, languages=("en", "zh-CN")) -> str:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            cleaned_transcript = " ".join(
                [entry["text"] for entry in transcript if entry["text"].lower() not in ["[music]"]]
            )
            return cleaned_transcript
        except Exception as e:
            logging.error(f"Error extracting YouTube transcript: {str(e)}")
            raise


@app.command()
def extract_content(
    urls: List[str],
    output_dir: str = "videos/transcripts/",
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transcriber = YouTubeTranscriber()
    for url in urls:
        try:
            video_id = url.split("v=")[-1]
            transcript = transcriber.extract_transcript(video_id)
            print("Transcript extracted successfully.")

            # Save transcript to file
            output_file = os.path.join(output_dir, f"{video_id}.txt")
            with open(output_file, "w") as file:
                file.write(transcript)

            print(f"Transcript saved to {output_file}")
            print("First 500 characters of the transcript:")
            print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")


@app.command()
def instruct_content(youtube_urls: List[str], language: str = "en") -> None:
    console = Console()
    extractor = YouTubeTranscriber()
    for url in youtube_urls:
        try:
            with console.status("[bold green]Processing URL...") as status:
                video_id = url.split("v=")[-1]
                content = extractor.extract_transcript(video_id)
                status.update("[bold blue]Generating Clips...")
                clips = table.extract_models(content, language=language)
                table.console_table(clips)

            console.print("\nChapter extraction complete!")
        except Exception as e:
            logging.error(f"An error occurred while processing {url}: {str(e)}")


r"""
# CUDA Mode Keynote | Andrej Karpathy | Eureka Labs
# A Hackers' Guide to Language Models
# Transformer论文逐段精读

python -m demo.content_parser.youtube_transcriber_instructor extract-content \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "https://www.youtube.com/watch?v=jkrNMKz9pWU" \
    "https://www.youtube.com/watch?v=nzqlFIcCSWQ"

python -m demo.content_parser.youtube_transcriber_instructor instruct-content \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "https://www.youtube.com/watch?v=jkrNMKz9pWU" \
    "https://www.youtube.com/watch?v=nzqlFIcCSWQ" \
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
