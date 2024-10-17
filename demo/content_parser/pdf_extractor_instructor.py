import re
import logging
import os
import unicodedata
from typing import List

import typer
import pymupdf
import instructor
from rich.console import Console
import google.generativeai as genai

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

app = typer.Typer()


class PDFExtractor:
    def extract_content(self, file_path: str) -> str:
        try:
            doc = pymupdf.open(file_path)
            content = " ".join(page.get_text() for page in doc)
            doc.close()

            # Normalize the text to handle special characters and remove accents
            normalized_content = unicodedata.normalize('NFKD', content)

            return normalized_content
        except Exception as e:
            logging.error(f"Error extracting PDF content: {str(e)}")
            raise


def get_pdf_file_name(file: str):
    pattern = r'([^/]+\.pdf)$'
    match = re.search(pattern, file)
    if match:
        pdf_file_name = match.group(1)
        return pdf_file_name
    else:
        raise Exception("must use *.pdf file")


@app.command()
def extract_content(
    pdf_files: List[str],
    output_dir: str = 'videos/transcripts/',
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extractor = PDFExtractor()
    for pdf_file in pdf_files:
        try:
            pdf_name = get_pdf_file_name(pdf_file)
            content = extractor.extract_content(pdf_file)
            print(f"PDF {pdf_file} content extracted successfully:")
            print(content[:500] + "..." if len(content) > 500 else content)
            # Save transcript to file
            output_file = os.path.join(output_dir, f"{pdf_name}.txt")
            with open(output_file, 'w') as file:
                file.write(content)
        except Exception as e:
            print(f"An error occurred: {str(e)}")


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
def instruct_content(test_urls: List[str], language: str = 'en') -> None:
    console = Console()
    extractor = PDFExtractor()
    for url in test_urls:
        try:
            with console.status("[bold green]Processing URL...") as status:
                content = extractor.extract_content(url)
                status.update("[bold blue]Generating Clips...")
                chapters = extract_chapters(content, language=language)
                chapter.console_table(chapters)

            console.print("\nChapter extraction complete!")
        except Exception as e:
            logging.error(f"An error occurred while processing {url}: {str(e)}")


r"""
python -m demo.content_parser.pdf_extractor_instructor extract-content \
    "/Users/wuyong/Documents/论文/llm/Attention Is All You Need.pdf" \
    "/Users/wuyong/Documents/论文/llm/《深度学习入门：基于Python的理论与实现》高清中文版.pdf"

python -m demo.content_parser.pdf_extractor_instructor instruct-content \
    "/Users/wuyong/Documents/论文/llm/Attention Is All You Need.pdf" \
    "/Users/wuyong/Documents/论文/llm/《深度学习入门：基于Python的理论与实现》高清中文版.pdf" \
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
