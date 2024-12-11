import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HTTPRequest:
    def __init__(self, max_retries=3, backoff_factor=0.5, backoff_jitter=0.0) -> None:
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,  # s
            backoff_jitter=backoff_jitter,
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"],
            respect_retry_after_header=True,  # add Retry-After in header
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.http_session = requests.Session()
        self.http_session.mount("http://", adapter)
        self.http_session.mount("https://", adapter)

    def get(self, url, **kwargs):
        response = self.http_session.get(url, **kwargs)
        return response

    def post(self, url, **kwargs):
        response = self.http_session.post(url, **kwargs)
        return response
