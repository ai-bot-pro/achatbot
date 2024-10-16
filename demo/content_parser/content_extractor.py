import os
import logging
from typing import List
from urllib.parse import urlparse

from dotenv import load_dotenv
from rich.console import Console
import google.generativeai as genai
import instructor
import typer

from .youtube_transcriber_instructor import YouTubeTranscriber
from .website_extractor_instructor import WebsiteExtractor
from .pdf_extractor_instructor import PDFExtractor, get_pdf_file_name
from .table import chapter
from . import types

# Load environment variables from .env file
load_dotenv(override=True)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",
    ),
    mode=instructor.Mode.GEMINI_JSON,
)

app = typer.Typer()


class ContentExtractor:
    def __init__(self, is_save=False, save_dir=""):
        self.youtube_transcriber = YouTubeTranscriber()
        self.website_extractor = WebsiteExtractor()
        self.pdf_extractor = PDFExtractor()
        self._save_dir = save_dir
        self._is_save = is_save
        self._file_name = ""
        if not os.path.exists(save_dir) and is_save:
            os.makedirs(save_dir)

    def is_url(self, source: str) -> bool:
        try:
            # If the source doesn't start with a scheme, add 'https://'
            if not source.startswith(('http://', 'https://')):
                source = 'https://' + source

            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def extract_content(self, source: str) -> str:
        res = self._extract_content(source)
        if self._is_save:
            output_file = os.path.join(self._save_dir, f"{self._file_name}.txt")
            with open(output_file, 'w') as file:
                file.write(res)
        return res

    def _extract_content(self, source: str) -> str:
        try:
            if self.is_url(source):
                if any(pattern in source for pattern in ["youtube.com", "youtu.be"]):
                    self._file_name = source.split("v=")[-1]
                    return self.youtube_transcriber.extract_transcript(source)
                else:
                    self._file_name = source.split("/")[-1] \
                        if source.split("/")[-1] else source.split("/")[-2]
                    return self.website_extractor.extract_content(source)
            elif source.lower().endswith('.pdf'):
                self._file_name = get_pdf_file_name(source)
                return self.pdf_extractor.extract_content(source)
            else:
                raise ValueError("Unsupported source type")
        except Exception as e:
            logging.error(f"Error extracting content from {source}: {str(e)}")
            raise


@app.command()
def extract_content(sources: str) -> None:
    extractor = ContentExtractor()

    for source in sources:
        try:
            print(f"Extracting content from: {source}")
            content = extractor.extract_content(source)
            print(f"Extracted content (first 500 characters):\n{content[:500]}...")
            print(f"Total length of extracted content: {len(content)} characters")
            print("-" * 50)

        except Exception as e:
            logging.error(f"An error occurred while processing {source}: {str(e)}")


def extract_chapters(content: str, language="en"):
    res = client.chat.completions.create_partial(
        response_model=chapter.Chapters,
        messages=[
            {
                "role": "system",
                "content": f"Analyze the given YouTube transcript and extract chapters. For each chapter, provide a start timestamp, end timestamp, title, and summary. Output language should be in {types.TO_LLM_LANGUAGE[language]}",
            },
            {"role": "user", "content": content},
        ],
    )
    return res


@app.command()
def instruct_content(sources: List[str], language: str = 'en') -> None:
    console = Console()
    extractor = PDFExtractor()
    for source in sources:
        try:
            with console.status("[bold green]Processing URL...") as status:
                content = extractor.extract_content(source)
                status.update("[bold blue]Generating Clips...")
                chapters = extract_chapters(content, language=language)
                chapter.console_table(chapters)

            console.print("\nChapter extraction complete!")
        except Exception as e:
            logging.error(f"An error occurred while processing {source}: {str(e)}")


r"""
python -m demo.content_parser.content_extractor_instructor extract-content \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "/Users/wuyong/Documents/论文/llm/Attention Is All You Need.pdf"

python -m demo.content_parser.content_extractor_instructor instruct-content \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "/Users/wuyong/Documents/论文/llm/Attention Is All You Need.pdf" \
    --language zh
"""

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s',
        handlers=[
            # logging.FileHandler("extractor.log"),
            logging.StreamHandler()
        ],
    )
    app()
