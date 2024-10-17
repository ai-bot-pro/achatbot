import os
import requests
import re
import html
import logging
from typing import List
from urllib.parse import urlparse

import typer
import instructor
from bs4 import BeautifulSoup
import google.generativeai as genai
from rich.console import Console

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


class WebsiteExtractor:
    def __init__(self):
        """
        Initialize the WebsiteExtractor.
        """
        self.unwanted_tags = []
        self.user_agent = 'Mozilla/5.0'
        self.timeout = 10
        self.remove_patterns = [
            '!\\[.*?\\]\\(.*?\\)',
            '\\[([^\\]]+)\\]\\([^\\)]+\\)',
            'https?://\\S+|www\\.\\S+',
        ]

    def extract_content(self, url: str) -> str:
        try:
            # Normalize the URL
            normalized_url = self.normalize_url(url)

            # Request the webpage
            headers = {'User-Agent': self.user_agent}
            response = requests.get(normalized_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Parse the page content with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

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
                f"An unexpected error occurred while extracting content from {url}: {str(e)}")
            raise Exception(
                f"An unexpected error occurred while extracting content from {url}: {str(e)}")

    def normalize_url(self, url: str) -> str:
        # If the URL doesn't start with a scheme, add 'https://'
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

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
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)

        # Remove extra newlines
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

        # Apply custom cleaning patterns from config
        for pattern in self.remove_patterns:
            cleaned_content = re.sub(pattern, '', cleaned_content)

        return cleaned_content.strip()


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
def instruct_content(test_urls: List[str], language: str = 'en') -> None:
    console = Console()
    extractor = WebsiteExtractor()
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
python -m demo.content_parser.website_extractor_instructor extract-content \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "https://weedge.github.io/post/paper/rag/rag-for-llms-a-survey/"

python -m demo.content_parser.website_extractor_instructor instruct-content \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "https://en.wikipedia.org/wiki/Gundam" \
    "https://weedge.github.io/post/paper/rag/rag-for-llms-a-survey/" \
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
