import httpx


class HTTPXClientWrapper:
    """Wrapper to return the singleton client where needed."""

    async_client: httpx.AsyncClient = None

    def start(self, timeout: float = 30.0):
        """Instantiate the client. Call from the FastAPI startup hook."""
        self.async_client = httpx.AsyncClient(timeout=timeout)

    async def stop(self):
        """Gracefully shutdown. Call from FastAPI shutdown hook."""
        await self.async_client.aclose()
        self.async_client = None

    def __call__(self):
        """Calling the instantiated HTTPXClientWrapper returns the wrapped singleton."""
        # Ensure we don't use it if not started / running
        assert self.async_client is not None
        return self.async_client
