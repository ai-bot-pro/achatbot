import requests
import logging
from datetime import datetime
import json
import time
import os

from src.common.http import HTTPRequest
from src.common.interface import ILlm
from .base import BaseLLM
from src.common.session import Session
from src.common.types import PersonalAIProxyArgs


class PersonalAIProxy(BaseLLM, ILlm):
    """
    personalai proxy llm service:
    TODO:
        vllm, triton etc.. inference services,
        support to load owner llm ckpt to inference
        now just use cloud api llm model inference services
    """

    TAG = "llm_personalai_proxy"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**PersonalAIProxyArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        import geocoder

        self.geo = geocoder.ip("me")
        self.args = PersonalAIProxyArgs(**args)
        self.requests = HTTPRequest(max_retries=self.args.max_retry_cn)

    def generate(self, session: Session):
        # @TODO: personalai proxy need use comletions openai api
        logging.info("generate use chat_completion")
        for item in self.chat_completion(session):
            yield item

    def chat_completion(self, session: Session):
        if self.args.llm_stream is False:
            res = self._chat(session)
            yield res
        else:
            # yield from self._chat_stream(session)
            yield from self._chat(session)

    def count_tokens(self, text: str | bytes):
        pass

    def _chat(self, session: Session):
        url = self.args.api_url
        payload = json.dumps(
            {
                "config": {
                    "chat": {
                        "api_base": self.args.openai_api_base_url,
                        "model": self.args.model_name,
                        "system_prompt": self.args.llm_chat_system,
                        "api_key": os.getenv("OPENAI_API_KEY", ""),
                        "temperature": self.args.llm_temperature,
                        "top_p": self.args.llm_top_p,
                        "max_tokens": self.args.llm_max_tokens,
                        "stream": False,
                        "stop": self.args.llm_stop,
                    },
                    "qianfan": {
                        "model": self.args.model_name,
                        "api_key": os.getenv("QIANFAN_API_KEY", ""),
                        "secret_key": os.getenv("QIANFAN_SECRET_KEY", ""),
                    },
                    "search": {
                        "search_name": self.args.func_search_name,
                        "search_api_key": os.getenv("SEARCH_API_KEY", ""),
                        "serper_api_key": os.getenv("SERPER_API_KEY", ""),
                        "search1_api_key": os.getenv("SEARCH1_API_KEY", ""),
                    },
                    "weather": {
                        "weather_name": self.args.func_weather_name,
                        "openweahtermap_api_key": os.getenv("OPENWEATHERMAP_API_KEY", ""),
                    },
                },
                "location": {
                    "latitude": self.geo.latlng[0],
                    "longitude": self.geo.latlng[1],
                },
                "chat_type": self.args.model_type,
                "chat_bot": self.args.chat_bot,
                "chat_id": session.ctx.client_id,
                "input": session.ctx.state["prompt"],
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        headers = {"Content-Type": "application/json"}

        try:
            logging.debug(f"payload:{payload}")
            response = self.requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # Raises for HTTP errors
            data = response.json()
            return data["response"]
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            return ""
        except Exception as err:
            logging.error(f"An error occurred: {err}")
            return ""

    def _chat_stream(self, session: Session):
        # !TODO: @weedge
        yield ""
