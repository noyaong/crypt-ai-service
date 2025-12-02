"""CoinMarketCap API 클라이언트"""

from datetime import datetime

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from crypto_ai.models import CryptoQuote


class CMCClient:
    """CoinMarketCap API 클라이언트"""

    BASE_URL = "https://pro-api.coinmarketcap.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "X-CMC_PRO_API_KEY": api_key,
            "Accept": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _request(self, endpoint: str, params: dict | None = None) -> dict:
        """API 요청 (재시도 로직 포함)"""
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params or {}, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_listings_latest(
        self, limit: int = 100, convert: str = "USD"
    ) -> list[CryptoQuote]:
        """최신 암호화폐 목록 조회"""
        data = self._request(
            "/v1/cryptocurrency/listings/latest",
            {"limit": limit, "convert": convert},
        )

        quotes = []
        for coin in data.get("data", []):
            quote = coin["quote"][convert]
            quotes.append(
                CryptoQuote(
                    symbol=coin["symbol"],
                    name=coin["name"],
                    price=quote["price"],
                    volume_24h=quote["volume_24h"],
                    market_cap=quote["market_cap"],
                    percent_change_1h=quote["percent_change_1h"] or 0,
                    percent_change_24h=quote["percent_change_24h"] or 0,
                    percent_change_7d=quote["percent_change_7d"] or 0,
                    last_updated=datetime.fromisoformat(
                        quote["last_updated"].replace("Z", "+00:00")
                    ),
                )
            )
        return quotes

    def get_quote(self, symbol: str, convert: str = "USD") -> CryptoQuote | None:
        """특정 코인 시세 조회"""
        try:
            data = self._request(
                "/v2/cryptocurrency/quotes/latest",
                {"symbol": symbol, "convert": convert},
            )
        except requests.HTTPError:
            return None

        coin_data = data.get("data", {}).get(symbol)
        if not coin_data:
            return None

        coin = coin_data[0] if isinstance(coin_data, list) else coin_data
        quote = coin["quote"][convert]

        return CryptoQuote(
            symbol=coin["symbol"],
            name=coin["name"],
            price=quote["price"],
            volume_24h=quote["volume_24h"],
            market_cap=quote["market_cap"],
            percent_change_1h=quote["percent_change_1h"] or 0,
            percent_change_24h=quote["percent_change_24h"] or 0,
            percent_change_7d=quote["percent_change_7d"] or 0,
            last_updated=datetime.fromisoformat(
                quote["last_updated"].replace("Z", "+00:00")
            ),
        )

    def get_global_metrics(self) -> dict:
        """글로벌 시장 지표"""
        data = self._request("/v1/global-metrics/quotes/latest")
        return data.get("data", {})

    def get_trending(self) -> list[dict]:
        """트렌딩 코인 목록 (유료 플랜 필요)"""
        try:
            data = self._request("/v1/cryptocurrency/trending/most-visited")
            return data.get("data", [])
        except requests.HTTPError:
            return []
