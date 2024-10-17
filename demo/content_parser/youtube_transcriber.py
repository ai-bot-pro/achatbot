import os
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
import logging
import typer

logger = logging.getLogger(__name__)
app = typer.Typer()


class YouTubeTranscriber:

    def extract_transcript(self, video_id: str, languages=('en', 'zh-CN')) -> str:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            cleaned_transcript = " ".join([
                entry['text'] for entry in transcript
                if entry['text'].lower() not in ["[music]"]
            ])
            return cleaned_transcript
        except Exception as e:
            logger.error(f"Error extracting YouTube transcript: {str(e)}")
            raise


@app.command()
def main(
    urls: List[str],
    output_dir: str = 'videos/transcripts/',
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
            with open(output_file, 'w') as file:
                file.write(transcript)

            print(f"Transcript saved to {output_file}")
            print("First 500 characters of the transcript:")
            print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")


r"""
# CUDA Mode Keynote | Andrej Karpathy | Eureka Labs
# A Hackers' Guide to Language Models
# Transformer论文逐段精读

python demo/content_parser/youtube_extractor.py \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "https://www.youtube.com/watch?v=jkrNMKz9pWU" \
    "https://www.youtube.com/watch?v=nzqlFIcCSWQ"
"""
if __name__ == "__main__":
    app()
