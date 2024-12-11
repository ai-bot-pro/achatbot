import os
import requests
import re
import html
import logging
from typing import List
from urllib.parse import urlparse

import typer
from bs4 import BeautifulSoup
from rich.console import Console

from .table import table

app = typer.Typer()


class WebsiteExtractor:
    def __init__(self):
        """
        Initialize the WebsiteExtractor.
        """
        self.unwanted_tags = []
        self.user_agent = "Mozilla/5.0"
        self.timeout = 10
        self.remove_patterns = [
            "!\\[.*?\\]\\(.*?\\)",
            "\\[([^\\]]+)\\]\\([^\\)]+\\)",
            "https?://\\S+|www\\.\\S+",
        ]

    def extract_content(self, url: str) -> str:
        try:
            # Normalize the URL
            normalized_url = self.normalize_url(url)

            # Request the webpage
            headers = {"User-Agent": self.user_agent}
            response = requests.get(normalized_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Parse the page content with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            self.remove_unwanted_elements(soup)

            # Extract and clean the text content
            raw_text = soup.get_text(separator="\n")  # Get all text content
            cleaned_content = self.clean_content(raw_text)

            return cleaned_content
        except requests.RequestException as e:
            logging.error(f"Failed to extract content from {url}: {str(e)}")
            raise Exception(f"Failed to extract content from {url}: {str(e)}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while extracting content from {url}: {str(e)}"
            )
            raise Exception(
                f"An unexpected error occurred while extracting content from {url}: {str(e)}"
            )

    def normalize_url(self, url: str) -> str:
        # If the URL doesn't start with a scheme, add 'https://'
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Parse the URL
        parsed = urlparse(url)

        # Ensure the URL has a valid scheme and netloc
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError(f"Invalid URL: {url}")

        return parsed.geturl()

    def remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """
        Remove unwanted elements from the BeautifulSoup object.

        Args:
                soup (BeautifulSoup): The BeautifulSoup object to clean.
        """
        for tag in self.unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()

    def clean_content(self, content: str) -> str:
        # Decode HTML entities
        cleaned_content = html.unescape(content)

        # Remove extra whitespace
        cleaned_content = re.sub(r"\s+", " ", cleaned_content)

        # Remove extra newlines
        cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)

        # Apply custom cleaning patterns from config
        for pattern in self.remove_patterns:
            cleaned_content = re.sub(pattern, "", cleaned_content)

        return cleaned_content.strip()


@app.command()
def extract_content(test_urls: List[str]) -> None:
    extractor = WebsiteExtractor()
    for url in test_urls:
        try:
            print(f"Extracting content from: {url}")
            content = extractor.extract_content(url)
            print(f"Extracted content (first 500 characters):\n{content[:500]}...")
            print(f"Total length of extracted content: {len(content)} characters")
            print("-" * 50)

        except Exception as e:
            logging.error(f"An error occurred while processing {url}: {str(e)}")


@app.command()
def instruct_content(
    test_urls: List[str],
    language: str = "en",
    mode="text",
) -> None:
    console = Console()
    extractor = WebsiteExtractor()
    for url in test_urls:
        try:
            with console.status("[bold green]Processing URL...") as status:
                content = extractor.extract_content(url)
                status.update("[bold blue]Generating Clips...")
                data_models = table.extract_models(content, mode=mode, language=language)
                table.console_table(data_models)

            console.print("\nChapter extraction complete!")
        except Exception as e:
            logging.error(f"An error occurred while processing {url}: {str(e)}", exc_info=True)


r"""
python -m demo.content_parser.website_extractor_instructor extract-content \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "https://weedge.github.io/post/paper/rag/rag-for-llms-a-survey/"

python -m demo.content_parser.website_extractor_instructor instruct-content \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "https://en.wikipedia.org/wiki/Gundam" \
    "https://weedge.github.io/post/paper/rag/rag-for-llms-a-survey/" \
    --language zh

TABLE_MODEL=podcast python -m demo.content_parser.website_extractor_instructor instruct-content \
    "https://en.wikipedia.org/wiki/Large_language_model" --language zh
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
