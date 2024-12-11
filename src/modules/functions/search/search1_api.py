import requests
import logging
import json
import os

from src.common.types import Search1ApiArgs
from .api import SearchBaseApi

from dotenv import load_dotenv

load_dotenv(override=True)


class Search1Api(SearchBaseApi):
    BASE_URL = "https://api.search1api.com/search"
    TAG = "search1_api"

    def __init__(self, **args) -> None:
        super().__init__()
        self.args = Search1ApiArgs(**args)

    def _web_search(self, session, query: str) -> str:
        api_key = os.getenv("SEARCH1_API_KEY", "")
        url = self.BASE_URL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "search_service": self.args.search_service,
            "image": self.args.image,
            "max_results": self.args.max_results,
            "crawl_results": self.args.crawl_results,
        }

        try:
            response = self.requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raises for HTTP errors
            data = response.json()
            snippets = [item["snippet"] for item in data["results"]]
            return json.dumps(snippets)
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return json.dumps({"error": "Failed to fetch search results"})
        except Exception as err:
            print(f"An error occurred: {err}")
            return json.dumps({"error": "Failed to fetch search results"})
