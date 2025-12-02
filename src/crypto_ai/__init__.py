"""
Crypto AI Analysis Service
MacBook MPS + PyTorch 기반 암호화폐 AI 분석 서비스
"""

from crypto_ai.client import CMCClient
from crypto_ai.analyzer import ChartAnalyzer, get_device
from crypto_ai.insight import OnChainInsightAnalyzer
from crypto_ai.models import CryptoQuote, TrendDirection
from crypto_ai.data_sources import (
    AlternativeMeClient,
    BinanceClient,
    CoinGeckoClient,
    UnifiedDataCollector,
)

__version__ = "0.1.0"
__all__ = [
    "CryptoAIService",
    "CMCClient",
    "ChartAnalyzer",
    "OnChainInsightAnalyzer",
    "CryptoQuote",
    "TrendDirection",
    "get_device",
    # 무료 데이터 소스
    "AlternativeMeClient",
    "BinanceClient",
    "CoinGeckoClient",
    "UnifiedDataCollector",
]


class CryptoAIService:
    """통합 암호화폐 AI 분석 서비스"""

    def __init__(self, api_key: str | None = None):
        import os
        from dotenv import load_dotenv

        load_dotenv()

        self.api_key = api_key or os.getenv("CMC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "CMC_API_KEY가 필요합니다. "
                ".env 파일에 설정하거나 생성자에 전달하세요."
            )

        self.cmc = CMCClient(self.api_key)
        self.chart_analyzer = ChartAnalyzer()
        self.insight_analyzer = OnChainInsightAnalyzer(self.cmc)

        device = get_device()
        print(f"🔧 CryptoAIService 초기화 완료 (Device: {device})")

    def get_price(self, symbol: str) -> dict:
        """시세 조회"""
        quote = self.cmc.get_quote(symbol.upper())
        if not quote:
            return {"error": f"{symbol} not found"}

        return {
            "symbol": quote.symbol,
            "name": quote.name,
            "price_usd": f"${quote.price:,.2f}",
            "change_1h": f"{quote.percent_change_1h:+.2f}%",
            "change_24h": f"{quote.percent_change_24h:+.2f}%",
            "change_7d": f"{quote.percent_change_7d:+.2f}%",
            "volume_24h": f"${quote.volume_24h:,.0f}",
            "market_cap": f"${quote.market_cap:,.0f}",
        }

    def analyze_chart(self, prices: list[float], volumes: list[float]) -> dict:
        """차트 기술적 분석"""
        import numpy as np

        prices_arr = np.array(prices)
        volumes_arr = np.array(volumes)
        return self.chart_analyzer.analyze(prices_arr, volumes_arr)

    def get_market_insights(self, limit: int = 50) -> dict:
        """시장 전체 인사이트"""
        quotes = self.cmc.get_listings_latest(limit=limit)
        sentiment = self.insight_analyzer.analyze_market_sentiment(quotes)
        overview = self.insight_analyzer.get_market_overview()

        # CryptoQuote를 dict로 변환
        def quote_to_dict(q: CryptoQuote) -> dict:
            return {
                "symbol": q.symbol,
                "name": q.name,
                "price": q.price,
                "change_24h": q.percent_change_24h,
            }

        return {
            "market_overview": overview,
            "sentiment_analysis": sentiment,
            "top_gainers": [
                quote_to_dict(q)
                for q in sorted(quotes, key=lambda x: x.percent_change_24h, reverse=True)[:5]
            ],
            "top_losers": [
                quote_to_dict(q)
                for q in sorted(quotes, key=lambda x: x.percent_change_24h)[:5]
            ],
        }

    def get_coin_insights(self, symbol: str) -> list[str]:
        """특정 코인 인사이트"""
        return self.insight_analyzer.generate_insights(symbol.upper())
