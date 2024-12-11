import re
import logging
import os
import unicodedata
from typing import List

import typer
import pymupdf
from rich.console import Console

from .table import table


app = typer.Typer()


class PDFExtractor:
    def extract_content(self, file_path: str) -> str:
        try:
            doc = pymupdf.open(file_path)
            content = " ".join(page.get_text() for page in doc)
            doc.close()

            # Normalize the text to handle special characters and remove accents
            normalized_content = unicodedata.normalize("NFKD", content)

            return normalized_content
        except Exception as e:
            logging.error(f"Error extracting PDF content: {str(e)}")
            raise


def get_pdf_file_name(file: str):
    pattern = r"([^/]+\.pdf)$"
    match = re.search(pattern, file)
    if match:
        pdf_file_name = match.group(1)
        return pdf_file_name
    else:
        raise Exception("must use *.pdf file")


@app.command()
def extract_content(
    pdf_files: List[str],
    output_dir: str = "videos/transcripts/",
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
            with open(output_file, "w") as file:
                file.write(content)
        except Exception as e:
            print(f"An error occurred: {str(e)}")


@app.command()
def instruct_content(test_urls: List[str], language: str = "en") -> None:
    console = Console()
    extractor = PDFExtractor()
    for url in test_urls:
        try:
            with console.status("[bold green]Processing URL...") as status:
                content = extractor.extract_content(url)
                status.update("[bold blue]Generating Clips...")
                clips = table.extract_models(content, language=language)
                table.console_table(clips)

            console.print("\nChapter extraction complete!")
        except Exception as e:
            logging.error(f"An error occurred while processing {url}: {str(e)}")


r"""
python -m demo.content_parser.pdf_extractor_instructor extract-content \
    "/Users/wuyong/Documents/论文/llm/Attention Is All You Need.pdf" \
    "/Users/wuyong/Documents/论文/llm/《深度学习入门：基于Python的理论与实现》高清中文版.pdf"

python -m demo.content_parser.pdf_extractor_instructor extract-content \
    "/Users/wuyong/Desktop/iOS_18_All_New_Features_Sept_2024.pdf"

python -m demo.content_parser.pdf_extractor_instructor instruct-content \
    "/Users/wuyong/Documents/论文/llm/Attention Is All You Need.pdf" \
    "/Users/wuyong/Documents/论文/llm/《深度学习入门：基于Python的理论与实现》高清中文版.pdf" \
    --language zh

python -m demo.content_parser.pdf_extractor_instructor instruct-content \
    "/Users/wuyong/Desktop/iOS_18_All_New_Features_Sept_2024.pdf" \
    --language zh

#TODO: use OCR model extract PDF content e.g.: OCR-GOT2.0 :)
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
