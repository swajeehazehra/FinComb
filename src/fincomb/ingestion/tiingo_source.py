"""Tiingo end-of-day prices: the fallback source.

Tiingo's free "Starter" plan needs an API token (no credit card):
sign up at https://www.tiingo.com, then copy the token from
https://www.tiingo.com/account/api/token into TIINGO_API_KEY in your .env.
Free-plan limits: 50 requests/hour, 1,000/day, 500 unique symbols/month.

The token is sent in the Authorization header, never in the URL, so it can't
leak into logs. Share-class tickers use dashes on Tiingo (BRK.B -> BRK-B).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from urllib.parse import quote, urlencode

import pandas as pd

from fincomb.exceptions import DataSourceAuthError, DataSourceError, SymbolNotFoundError
from fincomb.ingestion.base import MarketDataSource, with_retries

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices?{query}"
TOKEN_HELP = (
    "Tiingo needs a free API token: sign up at https://www.tiingo.com, copy your token "
    "from https://www.tiingo.com/account/api/token and set TIINGO_API_KEY=... in .env"
)


def _http_get_json(url: str, token: str, timeout: float = 30.0) -> object:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "User-Agent": "fincomb/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:200]
        if exc.code in (401, 403):
            raise DataSourceAuthError(
                f"Tiingo rejected the API token (HTTP {exc.code}). " + TOKEN_HELP
            ) from exc
        if exc.code == 404:
            raise SymbolNotFoundError(f"Tiingo: ticker not found ({body})") from exc
        if exc.code == 429:
            raise DataSourceError(
                "Tiingo rate limit reached (free plan: 50 requests/hour, "
                "1,000/day); try again later"
            ) from exc
        raise DataSourceError(f"Tiingo HTTP {exc.code}: {body}") from exc


class TiingoSource(MarketDataSource):
    name = "tiingo"

    def __init__(
        self,
        api_key: str | None,
        http_get_json: Callable[[str, str], object] = _http_get_json,
        attempts: int = 3,
        base_delay: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.http_get_json = http_get_json  # injectable so tests never touch the network
        self.attempts = attempts
        self.base_delay = base_delay

    @staticmethod
    def tiingo_ticker(symbol: str) -> str:
        return symbol.upper().replace(".", "-")

    def _fetch_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if not self.api_key:
            raise DataSourceAuthError("TIINGO_API_KEY is not set. " + TOKEN_HELP)
        query = urlencode(
            {"startDate": start.isoformat(), "endDate": end.isoformat(), "format": "json"}
        )
        url = TIINGO_URL.format(ticker=quote(self.tiingo_ticker(symbol)), query=query)

        def call() -> pd.DataFrame:
            payload = self.http_get_json(url, self.api_key)
            if isinstance(payload, dict):  # errors come back as {"detail": "..."}
                raise DataSourceError(f"Tiingo: {payload.get('detail', payload)}")
            if not payload:
                raise SymbolNotFoundError(f"Tiingo has no data for {symbol} in that range")
            df = pd.DataFrame(payload)
            # Raw OHLC plus Tiingo's split/dividend-adjusted close.
            return df[["date", "open", "high", "low", "close", "adjClose", "volume"]]

        return with_retries(
            call,
            attempts=self.attempts,
            base_delay=self.base_delay,
            what=f"tiingo download {symbol}",
        )
