import requests
import logging
import json
import os

from src.common.types import OpenWeatherMapArgs
from src.modules.functions.weather.api import WeatherBaseApi


class OpenWeatherMap(WeatherBaseApi):
    """
    - https://openweathermap.org/api/one-call-3#data
    - https://openweathermap.org/api/one-call-3#multi
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    TAG = "openweathermap_api"

    def __init__(self, **args) -> None:
        super().__init__()
        self.args = OpenWeatherMapArgs(**args)

    def _get_weather(self, session, longitude: float, latitude: float) -> str:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")
        url = f"{OpenWeatherMap.BASE_URL}?appid={api_key}&lat={latitude}&lon={longitude}&lang={self.args.lang}&units={self.args.units}"
        try:
            response = requests.get(url)
            # Raises a HTTPError if the HTTP request returned an unsuccessful status code
            response.raise_for_status()
            data = response.json()
            return json.dumps(data)
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            return json.dumps({"error": "Failed to fetch weather data"})
        except Exception as err:
            logging.error(f"An error occurred: {err}")
            return json.dumps({"error": "Failed to fetch weather data"})
