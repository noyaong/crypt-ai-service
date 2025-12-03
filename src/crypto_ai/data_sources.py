"""
무료 OHLCV 데이터 소스 클라이언트
- Binance API (무료, API 키 불필요)
- CoinGecko API (무료, rate limit: 30/min)
- Alternative.me API (무료, Fear & Greed Index)
  - API 문서: https://alternative.me/crypto/api/
  - Fear & Greed: https://alternative.me/crypto/fear-and-greed-index/#api
"""

import os
from datetime import datetime, timedelta
from typing import Callable, Literal
import time

import httpx
import pandas as pd
import numpy as np

from crypto_ai.models import CryptoQuote


# ============================================================
# Binance API Client (무료, 무제한 OHLCV)
# ============================================================

class BinanceClient:
    """
    Binance 공개 API 클라이언트
    - API 키 불필요
    - 2017년부터의 히스토리 데이터
    - 1분/5분/15분/1시간/4시간/1일 등 다양한 인터벌
    """

    BASE_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"

    INTERVALS = Literal[
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M"
    ]

    def __init__(self, futures: bool = False):
        self.futures = futures
        self.base_url = self.FUTURES_URL if futures else self.BASE_URL

    def get_symbols(self) -> list[str]:
        """사용 가능한 심볼 목록"""
        endpoint = "/fapi/v1/exchangeInfo" if self.futures else "/api/v3/exchangeInfo"
        with httpx.Client() as client:
            response = client.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            data = response.json()
        return [s["symbol"] for s in data["symbols"]]

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 조회 (단일 요청, 최대 1000개)

        Args:
            symbol: 거래쌍 (예: BTCUSDT, ETHUSDT)
            interval: 봉 간격 (1m, 5m, 15m, 1h, 4h, 1d 등)
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 최대 개수 (기본 1000, 최대 1000)

        Returns:
            DataFrame with columns: [open_time, open, high, low, close, volume, ...]
        """
        endpoint = "/fapi/v1/klines" if self.futures else "/api/v3/klines"

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        with httpx.Client() as client:
            response = client.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        # 타입 변환
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        return df

    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        progress_callback: Callable | None = None,
    ) -> pd.DataFrame:
        """
        대량 히스토리 데이터 조회 (자동 페이지네이션)

        Args:
            symbol: 거래쌍
            interval: 봉 간격
            start_time: 시작 시간 (기본: 1년 전)
            end_time: 종료 시간 (기본: 현재)
            progress_callback: 진행 상황 콜백 함수

        Returns:
            전체 기간의 OHLCV DataFrame
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=365)

        all_data = []
        current_start = start_time

        while current_start < end_time:
            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000,
            )

            if df.empty:
                break

            all_data.append(df)

            # 다음 배치 시작점
            last_time = df["close_time"].iloc[-1].to_pydatetime()
            if last_time <= current_start:
                break
            current_start = last_time + timedelta(milliseconds=1)

            if progress_callback:
                progress_callback(len(all_data), current_start)

            # Rate limit 방지
            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["open_time"])
        return result.sort_values("open_time").reset_index(drop=True)

    def get_24h_ticker(self, symbol: str | None = None) -> dict | list[dict]:
        """24시간 티커 정보"""
        endpoint = "/api/v3/ticker/24hr"
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()

        with httpx.Client() as client:
            response = client.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()


# ============================================================
# CoinGecko API Client (무료, OHLCV + 메타데이터)
# ============================================================

class CoinGeckoClient:
    """
    CoinGecko 공개 API 클라이언트
    - 무료 (30 calls/min)
    - OHLCV, 마켓데이터, 메타데이터
    - 13,000+ 코인 지원
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: Pro API 키 (선택, 없으면 무료 티어)
        """
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-cg-pro-api-key"] = api_key

    def _request(self, endpoint: str, params: dict | None = None) -> dict:
        """API 요청 (rate limit 자동 처리)"""
        with httpx.Client() as client:
            response = client.get(
                f"{self.BASE_URL}{endpoint}",
                params=params or {},
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == 429:
                # Rate limited - wait and retry
                time.sleep(60)
                response = client.get(
                    f"{self.BASE_URL}{endpoint}",
                    params=params or {},
                    headers=self.headers,
                    timeout=30,
                )

            response.raise_for_status()
            return response.json()

    def get_global_data(self) -> dict:
        """
        글로벌 암호화폐 시장 데이터 조회

        Returns:
            {
                "total_market_cap": {"usd": 1234567890, ...},
                "total_volume": {"usd": 123456789, ...},
                "market_cap_percentage": {"btc": 45.5, "eth": 18.2, ...},
                "market_cap_change_percentage_24h_usd": 1.23,
                "active_cryptocurrencies": 10000,
                ...
            }
        """
        result = self._request("/global")
        return result.get("data", {})

    def get_coin_list(self) -> list[dict]:
        """
        지원되는 모든 코인 목록

        Returns:
            [{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}, ...]
        """
        return self._request("/coins/list")

    def get_coin_markets(
        self,
        vs_currency: str = "usd",
        order: str = "market_cap_desc",
        per_page: int = 100,
        page: int = 1,
        sparkline: bool = False,
    ) -> list[dict]:
        """
        시가총액 순 코인 목록 + 시세

        Args:
            vs_currency: 기준 통화 (usd, krw, btc 등)
            order: 정렬 기준
            per_page: 페이지당 개수 (최대 250)
            page: 페이지 번호
            sparkline: 7일 스파크라인 포함 여부

        Returns:
            코인 목록 + 시세 정보
        """
        return self._request("/coins/markets", {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": str(sparkline).lower(),
        })

    def get_coin_ohlc(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int | str = 30,
    ) -> pd.DataFrame:
        """
        OHLC 데이터 조회

        Args:
            coin_id: CoinGecko 코인 ID (예: bitcoin, ethereum)
            vs_currency: 기준 통화
            days: 조회 기간 (1, 7, 14, 30, 90, 180, 365, "max")

        Returns:
            DataFrame [timestamp, open, high, low, close]

        Note:
            - 1-2일: 30분봉
            - 3-30일: 4시간봉
            - 31일+: 4일봉
        """
        data = self._request(f"/coins/{coin_id}/ohlc", {
            "vs_currency": vs_currency,
            "days": days,
        })

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def get_coin_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int | str = 30,
        interval: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        가격/시가총액/거래량 히스토리

        Args:
            coin_id: CoinGecko 코인 ID
            vs_currency: 기준 통화
            days: 조회 기간
            interval: 봉 간격 (daily만 지원, 유료는 hourly)

        Returns:
            {"prices": DataFrame, "market_caps": DataFrame, "volumes": DataFrame}
        """
        params = {
            "vs_currency": vs_currency,
            "days": days,
        }
        if interval:
            params["interval"] = interval

        data = self._request(f"/coins/{coin_id}/market_chart", params)

        result = {}
        for key in ["prices", "market_caps", "total_volumes"]:
            if key in data:
                df = pd.DataFrame(data[key], columns=["timestamp", key.rstrip("s")])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                result[key] = df

        return result

    def get_simple_price(
        self,
        ids: list[str],
        vs_currencies: list[str] = ["usd"],
        include_24hr_change: bool = True,
        include_market_cap: bool = True,
        include_24hr_vol: bool = True,
    ) -> dict:
        """
        간단 시세 조회 (빠름)

        Args:
            ids: 코인 ID 목록
            vs_currencies: 기준 통화 목록

        Returns:
            {"bitcoin": {"usd": 45000, "usd_24h_change": -1.5, ...}, ...}
        """
        return self._request("/simple/price", {
            "ids": ",".join(ids),
            "vs_currencies": ",".join(vs_currencies),
            "include_24hr_change": str(include_24hr_change).lower(),
            "include_market_cap": str(include_market_cap).lower(),
            "include_24hr_vol": str(include_24hr_vol).lower(),
        })

    def search(self, query: str) -> dict:
        """코인/거래소 검색"""
        return self._request("/search", {"query": query})


# ============================================================
# Alternative.me API Client (무료, Fear & Greed Index)
# ============================================================

class AlternativeMeClient:
    """
    Alternative.me 공개 API 클라이언트
    - 무료 (60 calls/min)
    - Fear & Greed Index
    - 문서: https://alternative.me/crypto/api/
    """

    BASE_URL = "https://api.alternative.me"

    def get_fear_greed_index(
        self,
        limit: int = 1,
        date_format: Literal["us", "cn", "kr", "world"] | None = None,
    ) -> dict:
        """
        Fear & Greed Index 조회

        Args:
            limit: 결과 개수 (1=최신, 0=전체 히스토리)
            date_format: 날짜 형식 (us=MM/DD/YYYY, cn/kr=YYYY/MM/DD, world=DD/MM/YYYY)

        Returns:
            {
                "value": 24,
                "value_classification": "Extreme Fear",
                "timestamp": datetime,
                "time_until_update": 2031
            }

        Note:
            - 0-25: Extreme Fear (극도의 공포)
            - 26-46: Fear (공포)
            - 47-52: Neutral (중립)
            - 53-74: Greed (탐욕)
            - 75-100: Extreme Greed (극도의 탐욕)
        """
        params = {"limit": limit}
        if date_format:
            params["date_format"] = date_format

        with httpx.Client() as client:
            response = client.get(f"{self.BASE_URL}/fng/", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

        if data.get("metadata", {}).get("error"):
            raise ValueError(f"API Error: {data['metadata']['error']}")

        result = data.get("data", [])
        if not result:
            return {}

        # 단일 결과
        if limit == 1:
            item = result[0]
            return {
                "value": int(item["value"]),
                "value_classification": item["value_classification"],
                "timestamp": datetime.fromtimestamp(int(item["timestamp"])),
                "time_until_update": int(item.get("time_until_update", 0)),
            }

        # 다중 결과
        return {
            "data": [
                {
                    "value": int(item["value"]),
                    "value_classification": item["value_classification"],
                    "timestamp": datetime.fromtimestamp(int(item["timestamp"])),
                }
                for item in result
            ]
        }

    def get_fear_greed_history(self, days: int = 30) -> pd.DataFrame:
        """
        Fear & Greed Index 히스토리 조회

        Args:
            days: 조회 일수

        Returns:
            DataFrame [timestamp, value, classification]
        """
        result = self.get_fear_greed_index(limit=days)
        data = result.get("data", [])

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.rename(columns={"value_classification": "classification"})
        return df.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def interpret_fear_greed(value: int) -> str:
        """Fear & Greed 값 해석"""
        if value <= 25:
            return "극도의 공포 (Extreme Fear)"
        elif value <= 46:
            return "공포 (Fear)"
        elif value <= 52:
            return "중립 (Neutral)"
        elif value <= 74:
            return "탐욕 (Greed)"
        else:
            return "극도의 탐욕 (Extreme Greed)"


# ============================================================
# 심볼 매핑 유틸리티
# ============================================================

# CoinGecko ID <-> Symbol 매핑 (주요 코인)
COINGECKO_ID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "LTC": "litecoin",
    "ETC": "ethereum-classic",
    "XLM": "stellar",
    "NEAR": "near",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
}


def symbol_to_coingecko_id(symbol: str) -> str | None:
    """심볼을 CoinGecko ID로 변환"""
    return COINGECKO_ID_MAP.get(symbol.upper())


def symbol_to_binance_pair(symbol: str, quote: str = "USDT") -> str:
    """심볼을 Binance 거래쌍으로 변환"""
    return f"{symbol.upper()}{quote}"


# ============================================================
# 통합 데이터 수집기
# ============================================================

class UnifiedDataCollector:
    """
    여러 소스에서 데이터를 수집하는 통합 클래스

    사용 예:
        collector = UnifiedDataCollector()

        # Binance에서 1시간봉 수집
        df = collector.get_ohlcv("BTC", source="binance", interval="1h", days=30)

        # CoinGecko에서 일봉 수집
        df = collector.get_ohlcv("BTC", source="coingecko", days=90)
    """

    def __init__(self):
        self.binance = BinanceClient()
        self.coingecko = CoinGeckoClient()

    def get_ohlcv(
        self,
        symbol: str,
        source: Literal["binance", "coingecko"] = "binance",
        interval: str = "1h",
        days: int = 30,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 수집

        Args:
            symbol: 코인 심볼 (BTC, ETH, ...)
            source: 데이터 소스
            interval: 봉 간격 (binance만 지원)
            days: 조회 기간
            end_time: 종료 시간

        Returns:
            통합 형식의 OHLCV DataFrame
        """
        if end_time is None:
            end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        if source == "binance":
            pair = symbol_to_binance_pair(symbol)
            df = self.binance.get_historical_klines(
                symbol=pair,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
            )

            if df.empty:
                return df

            # 통합 형식으로 변환
            return pd.DataFrame({
                "timestamp": df["open_time"],
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"],
            })

        elif source == "coingecko":
            coin_id = symbol_to_coingecko_id(symbol)
            if not coin_id:
                raise ValueError(f"Unknown symbol: {symbol}")

            df = self.coingecko.get_coin_ohlc(coin_id, days=days)

            # CoinGecko는 볼륨을 별도로 가져와야 함
            chart = self.coingecko.get_coin_market_chart(coin_id, days=days)
            volumes = chart.get("total_volumes", pd.DataFrame())

            if not volumes.empty:
                # 타임스탬프 기준 병합
                df = df.merge(
                    volumes.rename(columns={"total_volume": "volume"}),
                    on="timestamp",
                    how="left",
                )

            return df

        else:
            raise ValueError(f"Unknown source: {source}")

    def get_price(self, symbol: str) -> dict:
        """현재 시세 (CoinGecko)"""
        coin_id = symbol_to_coingecko_id(symbol)
        if not coin_id:
            # Binance fallback
            ticker = self.binance.get_24h_ticker(symbol_to_binance_pair(symbol))
            return {
                "symbol": symbol,
                "price": float(ticker["lastPrice"]),
                "change_24h": float(ticker["priceChangePercent"]),
                "volume_24h": float(ticker["quoteVolume"]),
            }

        data = self.coingecko.get_simple_price([coin_id])
        coin_data = data.get(coin_id, {})

        return {
            "symbol": symbol,
            "price": coin_data.get("usd", 0),
            "change_24h": coin_data.get("usd_24h_change", 0),
            "volume_24h": coin_data.get("usd_24h_vol", 0),
            "market_cap": coin_data.get("usd_market_cap", 0),
        }


# ============================================================
# CryptoPanic API Client (무료, 뉴스/소셜 센티멘트)
# ============================================================

class CryptoPanicClient:
    """
    CryptoPanic 공개 API 클라이언트
    - 무료 (API 키 필요, https://cryptopanic.com/developers/api/)
    - 암호화폐 뉴스 및 소셜 미디어 센티멘트
    - API 키 없이도 제한적 사용 가능
    """

    BASE_URL = "https://cryptopanic.com/api/developer/v2"

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: CryptoPanic API 키 (선택, 없으면 제한적 기능)
        """
        self.api_key = api_key

    def _request(self, endpoint: str, params: dict | None = None) -> dict:
        """API 요청"""
        params = params or {}
        if self.api_key:
            params["auth_token"] = self.api_key

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
        }

        with httpx.Client(headers=headers) as client:
            response = client.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

    def get_posts(
        self,
        currencies: str | list[str] | None = None,
        filter: Literal["rising", "hot", "bullish", "bearish", "important", "saved", "lol"] | None = None,
        kind: Literal["news", "media", "all"] = "news",
        regions: str = "en",
        public: bool = True,
        limit: int = 20,
    ) -> list[dict]:
        """
        뉴스/미디어 포스트 조회

        Args:
            currencies: 코인 심볼 (예: "BTC" 또는 ["BTC", "ETH"])
            filter: 필터 옵션
                - rising: 상승 중인 뉴스
                - hot: 핫한 뉴스
                - bullish: 긍정적 뉴스
                - bearish: 부정적 뉴스
                - important: 중요 뉴스
            kind: 종류 (news, media, all) - 기본값 news
            regions: 지역 필터 (en, ko, de, es, fr 등) - 기본값 en
            public: 공개 API 사용 여부
            limit: 결과 개수

        Returns:
            [
                {
                    "id": 123,
                    "title": "...",
                    "published_at": "2024-01-01T12:00:00Z",
                    "source": {"title": "CoinDesk", "domain": "coindesk.com"},
                    "currencies": [{"code": "BTC", "title": "Bitcoin"}],
                    "votes": {"positive": 10, "negative": 2, "important": 5, "liked": 3, ...},
                    ...
                },
                ...
            ]
        """
        params = {
            "public": str(public).lower(),
            "kind": kind,
            "regions": regions,
        }

        if currencies:
            if isinstance(currencies, list):
                currencies = ",".join(currencies)
            params["currencies"] = currencies.upper()

        if filter:
            params["filter"] = filter

        result = self._request("/posts/", params)
        posts = result.get("results", [])

        # limit 적용
        return posts[:limit]

    def get_sentiment_summary(
        self,
        symbol: str,
        limit: int = 30,
    ) -> dict:
        """
        특정 코인의 뉴스 센티멘트 요약

        Args:
            symbol: 코인 심볼 (예: BTC, ETH, AVAX)
            limit: 분석할 뉴스 개수

        Returns:
            {
                "symbol": "BTC",
                "total_posts": 30,
                "bullish_count": 10,
                "bearish_count": 5,
                "important_count": 8,
                "hot_count": 12,
                "sentiment_score": 0.2,
                "recent_headlines": [...],
            }
        """
        # 전체 뉴스 가져오기 (news only, 영어)
        all_posts = self.get_posts(currencies=symbol, kind="news", regions="en", limit=limit)

        # 다양한 필터로 뉴스 가져오기
        bullish_posts = self.get_posts(currencies=symbol, filter="bullish", kind="news", limit=limit)
        bearish_posts = self.get_posts(currencies=symbol, filter="bearish", kind="news", limit=limit)
        important_posts = self.get_posts(currencies=symbol, filter="important", kind="news", limit=limit)
        hot_posts = self.get_posts(currencies=symbol, filter="hot", kind="news", limit=limit)

        total = len(all_posts)
        bullish_count = len(bullish_posts)
        bearish_count = len(bearish_posts)
        important_count = len(important_posts)
        hot_count = len(hot_posts)

        # 센티멘트 점수 계산 (-1 ~ +1)
        if total > 0:
            sentiment_score = (bullish_count - bearish_count) / total
            bullish_ratio = bullish_count / total
            bearish_ratio = bearish_count / total
        else:
            sentiment_score = 0.0
            bullish_ratio = 0.0
            bearish_ratio = 0.0

        # bullish/bearish post ID 집합
        bullish_ids = {p.get("id") for p in bullish_posts}
        bearish_ids = {p.get("id") for p in bearish_posts}
        important_ids = {p.get("id") for p in important_posts}

        # 최근 헤드라인 (중요 뉴스 우선)
        def get_sentiment(post):
            pid = post.get("id")
            if pid in bullish_ids:
                return "bullish"
            elif pid in bearish_ids:
                return "bearish"
            return "neutral"

        # important 뉴스를 먼저, 그 다음 최신 뉴스
        sorted_posts = sorted(
            all_posts,
            key=lambda p: (p.get("id") not in important_ids, p.get("published_at", "")),
            reverse=True
        )

        recent_headlines = [
            {
                "title": post.get("title", ""),
                "source": post.get("source", {}).get("title", "Unknown"),
                "published_at": post.get("published_at", ""),
                "sentiment": get_sentiment(post),
                "is_important": post.get("id") in important_ids,
                "is_hot": post.get("id") in {p.get("id") for p in hot_posts},
            }
            for post in sorted_posts[:10]
        ]

        return {
            "symbol": symbol.upper(),
            "total_posts": total,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "important_count": important_count,
            "hot_count": hot_count,
            "bullish_ratio": round(bullish_ratio, 3),
            "bearish_ratio": round(bearish_ratio, 3),
            "sentiment_score": round(sentiment_score, 3),
            "recent_headlines": recent_headlines,
        }


# ============================================================
# LunarCrush Alternative (무료 소셜 센티멘트)
# ============================================================

class SocialSentimentCollector:
    """
    소셜 미디어 센티멘트 수집기 (CoinGecko 활용)
    - CoinGecko의 community_data 및 developer_data 활용
    - 추가 비용 없이 기본적인 소셜 지표 수집
    """

    def __init__(self):
        self.coingecko = CoinGeckoClient()

    def get_social_metrics(self, symbol: str) -> dict | None:
        """
        소셜 미디어 지표 수집

        Args:
            symbol: 코인 심볼

        Returns:
            {
                "symbol": "BTC",
                "twitter_followers": 5000000,
                "reddit_subscribers": 4000000,
                "reddit_active_users": 5000,
                "telegram_users": 100000,
                "github_stars": 70000,
                "github_forks": 35000,
                "github_commits_4w": 100,
                "social_score": 85,  # 0-100
            }
        """
        coin_id = symbol_to_coingecko_id(symbol.upper())
        if not coin_id:
            return None

        try:
            # CoinGecko coin detail API
            data = self.coingecko._request(f"/coins/{coin_id}", {
                "localization": "false",
                "tickers": "false",
                "market_data": "false",
                "community_data": "true",
                "developer_data": "true",
            })
        except Exception:
            return None

        community = data.get("community_data", {})
        developer = data.get("developer_data", {})

        # 소셜 점수 계산 (정규화된 지표들의 가중 평균)
        twitter = community.get("twitter_followers", 0) or 0
        reddit_subs = community.get("reddit_subscribers", 0) or 0
        reddit_active = community.get("reddit_accounts_active_48h", 0) or 0
        telegram = community.get("telegram_channel_user_count", 0) or 0
        github_stars = developer.get("stars", 0) or 0

        # 간단한 소셜 점수 (로그 스케일로 정규화)
        def log_score(value: int, max_expected: int) -> float:
            if value <= 0:
                return 0
            return min(100, (np.log10(value + 1) / np.log10(max_expected + 1)) * 100)

        social_score = (
            log_score(twitter, 10_000_000) * 0.3 +
            log_score(reddit_subs, 5_000_000) * 0.25 +
            log_score(reddit_active, 50_000) * 0.2 +
            log_score(telegram, 500_000) * 0.15 +
            log_score(github_stars, 100_000) * 0.1
        )

        return {
            "symbol": symbol.upper(),
            "twitter_followers": twitter,
            "reddit_subscribers": reddit_subs,
            "reddit_active_users": reddit_active,
            "telegram_users": telegram,
            "github_stars": github_stars,
            "github_forks": developer.get("forks", 0) or 0,
            "github_commits_4w": developer.get("commit_count_4_weeks", 0) or 0,
            "social_score": round(social_score, 1),
        }


# ============================================================
# 통합 센티멘트 수집기
# ============================================================

class SentimentAggregator:
    """
    여러 소스에서 센티멘트 데이터를 수집하는 통합 클래스

    사용 예:
        aggregator = SentimentAggregator()
        sentiment = aggregator.get_comprehensive_sentiment("BTC")
    """

    def __init__(self, cryptopanic_api_key: str | None = None):
        # 환경변수에서 API 키 로드
        if cryptopanic_api_key is None:
            cryptopanic_api_key = os.environ.get("CRYPTOPANIC_API_KEY")

        self.fear_greed = AlternativeMeClient()
        self.news = CryptoPanicClient(api_key=cryptopanic_api_key)
        self.social = SocialSentimentCollector()

    def get_comprehensive_sentiment(self, symbol: str) -> dict:
        """
        종합 센티멘트 데이터 수집

        Args:
            symbol: 코인 심볼

        Returns:
            {
                "symbol": "BTC",
                "fear_greed": {...},
                "news_sentiment": {...},
                "social_metrics": {...},
                "overall_sentiment": "bullish" | "bearish" | "neutral",
                "sentiment_score": 0.65,  # -1 ~ +1
            }
        """
        result = {
            "symbol": symbol.upper(),
            "fear_greed": None,
            "news_sentiment": None,
            "social_metrics": None,
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
        }

        scores = []
        weights = []

        # 1. Fear & Greed Index
        try:
            fng = self.fear_greed.get_fear_greed_index()
            result["fear_greed"] = fng
            # 0-100을 -1~+1로 변환
            fng_score = (fng.get("value", 50) - 50) / 50
            scores.append(fng_score)
            weights.append(0.3)
        except Exception:
            pass

        # 2. 뉴스 센티멘트
        try:
            news = self.news.get_sentiment_summary(symbol)
            result["news_sentiment"] = news
            scores.append(news.get("sentiment_score", 0))
            weights.append(0.4)
        except Exception:
            pass

        # 3. 소셜 지표
        try:
            social = self.social.get_social_metrics(symbol)
            result["social_metrics"] = social
            if social:
                # 소셜 점수를 -1~+1로 변환 (50 기준)
                social_score = (social.get("social_score", 50) - 50) / 50
                scores.append(social_score)
                weights.append(0.3)
        except Exception:
            pass

        # 종합 점수 계산
        if scores and weights:
            total_weight = sum(weights[:len(scores)])
            weighted_sum = sum(s * w for s, w in zip(scores, weights[:len(scores)]))
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0
            result["sentiment_score"] = round(overall_score, 3)

            if overall_score > 0.2:
                result["overall_sentiment"] = "bullish"
            elif overall_score < -0.2:
                result["overall_sentiment"] = "bearish"
            else:
                result["overall_sentiment"] = "neutral"

        return result


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    # Binance 테스트
    print("=== Binance API ===")
    binance = BinanceClient()

    # 심볼 목록
    symbols = binance.get_symbols()
    print(f"Total symbols: {len(symbols)}")

    # BTC 1시간봉 (최근 100개)
    df = binance.get_klines("BTCUSDT", interval="1h", limit=100)
    print(f"\nBTC/USDT 1h klines: {len(df)} rows")
    print(df.tail())

    # CoinGecko 테스트
    print("\n=== CoinGecko API ===")
    coingecko = CoinGeckoClient()

    # 시가총액 상위 10개
    markets = coingecko.get_coin_markets(per_page=10)
    print("\nTop 10 by market cap:")
    for coin in markets[:5]:
        print(f"  {coin['symbol'].upper()}: ${coin['current_price']:,.2f}")

    # BTC OHLC
    ohlc = coingecko.get_coin_ohlc("bitcoin", days=7)
    print(f"\nBTC OHLC (7d): {len(ohlc)} rows")
    print(ohlc.tail())

    # 통합 수집기 테스트
    print("\n=== Unified Collector ===")
    collector = UnifiedDataCollector()

    # Binance에서 수집
    df_binance = collector.get_ohlcv("BTC", source="binance", interval="4h", days=7)
    print(f"Binance BTC 4h: {len(df_binance)} rows")

    # CoinGecko에서 수집
    df_cg = collector.get_ohlcv("ETH", source="coingecko", days=30)
    print(f"CoinGecko ETH: {len(df_cg)} rows")
