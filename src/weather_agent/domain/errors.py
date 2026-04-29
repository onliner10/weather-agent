from __future__ import annotations


class WeatherProviderError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        self.message = message
        super().__init__(f"[{provider}] {message}")


class WeatherProviderUnavailableError(WeatherProviderError):
    def __init__(self, provider: str, message: str = "Provider is unavailable") -> None:
        super().__init__(provider, message)


class WeatherProviderTimeoutError(WeatherProviderError):
    def __init__(self, provider: str, message: str = "Provider request timed out") -> None:
        super().__init__(provider, message)


class WeatherProviderResponseError(WeatherProviderError):
    def __init__(
        self, provider: str, message: str = "Provider returned an invalid response"
    ) -> None:
        super().__init__(provider, message)
