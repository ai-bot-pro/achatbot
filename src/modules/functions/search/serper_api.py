import requests
import json
import os

import logging

from src.common.types import SerperApiArgs
from .api import SearchBaseApi

from dotenv import load_dotenv

load_dotenv(override=True)


class SerperApi(SearchBaseApi):
    BASE_URL = "https://google.serper.dev/search"
    TAG = "serper_api"

    def __init__(self, **args) -> None:
        super().__init__()
        self.args = SerperApiArgs(**args)

    def _web_search(self, session, query: str) -> str:
        api_key = os.getenv("SERPER_API_KEY", "")
        url = self.BASE_URL
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "gl": self.args.gl,
            "hl": self.args.hl,
            "page": self.args.page,
            "num": self.args.num,
        }

        try:
            response = self.requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raises for HTTP errors
            data = response.json()
            snippets = [item["snippet"] for item in data["organic"]]
            return json.dumps(snippets)
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return json.dumps({"error": "Failed to fetch search results"})
        except Exception as err:
            print(f"An error occurred: {err}")
            return json.dumps({"error": "Failed to fetch search results"})
