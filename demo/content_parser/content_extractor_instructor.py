import os
import logging
from typing import List
from urllib.parse import urlparse

from rich.console import Console
import typer

from .youtube_transcriber_instructor import YouTubeTranscriber
from .website_extractor_instructor import WebsiteExtractor
from .pdf_extractor_instructor import PDFExtractor, get_pdf_file_name

app = typer.Typer()


class ContentExtractor:
    def __init__(self, is_save=False, save_dir=""):
        self.youtube_transcriber = YouTubeTranscriber()
        self.website_extractor = WebsiteExtractor()
        self.pdf_extractor = PDFExtractor()
        self._save_dir = save_dir
        self._is_save = is_save
        self._file_name = ""
        if not os.path.exists(save_dir) and is_save and save_dir:
            os.makedirs(save_dir)

    def is_url(self, source: str) -> bool:
        try:
            # If the source doesn't start with a scheme, add 'https://'
            if not source.startswith(("http://", "https://")):
                source = "https://" + source

            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def extract_content(self, source: str) -> str:
        res = self._extract_content(source)
        if self._is_save:
            output_file = os.path.join(self._save_dir, f"{self._file_name}.txt")
            with open(output_file, "w") as file:
                file.write(res)
                logging.info(f"save to file: {output_file}")
        return res

    def _extract_content(self, source: str) -> str:
        try:
            if self.is_url(source):
                if any(pattern in source for pattern in ["youtube.com", "youtu.be"]):
                    self._file_name = source.split("v=")[-1]
                    return self.youtube_transcriber.extract_transcript(self._file_name)
                else:
                    self._file_name = (
                        source.split("/")[-1] if source.split("/")[-1] else source.split("/")[-2]
                    )
                    return self.website_extractor.extract_content(source)
            elif source.lower().endswith(".pdf"):
                self._file_name = get_pdf_file_name(source)
                return self.pdf_extractor.extract_content(source)
            else:
                raise ValueError("Unsupported source type")
        except Exception as e:
            logging.error(f"Error extracting content from {source}: {str(e)}")
            raise

    @property
    def file_name(self):
        return self._file_name


@app.command()
def extract_content(
    sources: List[str],
    is_save: bool = False,
    save_dir: str = "videos/transcripts/",
) -> None:
    extractor = ContentExtractor(is_save=is_save, save_dir=save_dir)

    for source in sources:
        try:
            print(f"Extracting content from: {source}")
            content = extractor.extract_content(source)
            print(f"Extracted content (first 500 characters):\n{content[:500]}...")
            print(f"Total length of extracted content: {len(content)} characters")
            print("-" * 70)

        except Exception as e:
            logging.error(f"An error occurred while processing {source}: {str(e)}", exc_info=True)


@app.command()
def instruct_content(
    sources: List[str],
    language: str = "en",
    mode: str = "text",
) -> None:
    from .table import table

    console = Console()
    extractor = ContentExtractor()
    for source in sources:
        try:
            with console.status("[bold green]Processing URL...") as status:
                content = extractor.extract_content(source)
                status.update("[bold blue]Generating Clips...")
                data_models = table.extract_models(content, mode=mode, language=language)
                table.console_table(data_models)

            console.print("\nChapter extraction complete!")
        except Exception as e:
            logging.error(f"An error occurred while processing {source}: {str(e)}", exc_info=True)


r"""
python -m demo.content_parser.content_extractor_instructor extract-content \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "/Users/wuyong/Documents/论文/llm/Attention Is All You Need.pdf"

python -m demo.content_parser.content_extractor_instructor extract-content --is-save \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "/Users/wuyong/Documents/论文/llm/Attention Is All You Need.pdf"


python -m demo.content_parser.content_extractor_instructor instruct-content \
    --language zh \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "/Users/wuyong/Documents/论文/llm/Attention Is All You Need.pdf"
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
