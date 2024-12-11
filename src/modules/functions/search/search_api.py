import requests
import logging
import json
import os


from src.common.types import SearchApiArgs
from .api import SearchBaseApi

from dotenv import load_dotenv

load_dotenv(override=True)


class SearchApi(SearchBaseApi):
    BASE_URL = "https://www.searchapi.io/api/v1/search"
    TAG = "search_api"

    def __init__(self, **args) -> None:
        super().__init__()
        self.args = SearchApiArgs(**args)

    def _web_search(self, session, query: str) -> str:
        api_key = os.getenv("SEARCH_API_KEY", "")
        params = {
            "engine": self.args.engine,
            "api_key": api_key,
            "q": query,
            "gl": self.args.gl,
            "hl": self.args.hl,
            "page": self.args.page,
            "num": self.args.num,
        }

        try:
            response = self.requests.get(self.BASE_URL, params=params)
            response.raise_for_status()  # Raises for HTTP errors
            data = response.json()
            snippets = [item["snippet"] for item in data["organic_results"]]
            return f"{json.dumps(snippets)}"
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return json.dumps({"error": "Failed to fetch search results"})
        except Exception as err:
            print(f"An error occurred: {err}")
            return json.dumps({"error": "Failed to fetch search results"})
