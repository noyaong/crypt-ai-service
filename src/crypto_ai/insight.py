"""온체인 데이터 기반 인사이트 분석"""

import numpy as np

from crypto_ai.client import CMCClient
from crypto_ai.data_sources import AlternativeMeClient
from crypto_ai.models import CryptoQuote


class OnChainInsightAnalyzer:
    """온체인 데이터 기반 인사이트 생성"""

    def __init__(self, cmc_client: CMCClient):
        self.cmc = cmc_client
        self.alternative_me = AlternativeMeClient()

    def analyze_market_sentiment(self, quotes: list[CryptoQuote]) -> dict:
        """시장 센티먼트 분석"""
        # Fear & Greed Index from Alternative.me API
        fear_greed_data = self._get_fear_greed_index()
        fear_greed = fear_greed_data.get("value", 50)
        sentiment = fear_greed_data.get("classification", "데이터 없음")

        if not quotes:
            return {
                "positive_ratio": 0,
                "avg_change_24h": 0,
                "avg_change_7d": 0,
                "fear_greed_index": fear_greed,
                "sentiment": sentiment,
            }

        positive_24h = sum(1 for q in quotes if q.percent_change_24h > 0)
        total = len(quotes)

        avg_change_24h = float(np.mean([q.percent_change_24h for q in quotes]))
        avg_change_7d = float(np.mean([q.percent_change_7d for q in quotes]))

        return {
            "positive_ratio": positive_24h / total if total > 0 else 0,
            "avg_change_24h": avg_change_24h,
            "avg_change_7d": avg_change_7d,
            "fear_greed_index": fear_greed,
            "sentiment": sentiment,
        }

    def _get_fear_greed_index(self) -> dict:
        """Alternative.me API에서 Fear & Greed Index 조회"""
        try:
            data = self.alternative_me.get_fear_greed_index(limit=1)
            return {
                "value": data.get("value", 50),
                "classification": data.get("value_classification", "Unknown"),
                "timestamp": data.get("timestamp"),
            }
        except Exception:
            # API 실패 시 기본값 반환
            return {
                "value": 50,
                "classification": "데이터 없음",
                "timestamp": None,
            }

    def get_market_overview(self) -> dict:
        """시장 전체 현황"""
        global_metrics = self.cmc.get_global_metrics()
        quote = global_metrics.get("quote", {}).get("USD", {})

        return {
            "total_market_cap": quote.get("total_market_cap", 0),
            "total_volume_24h": quote.get("total_volume_24h", 0),
            "btc_dominance": global_metrics.get("btc_dominance", 0),
            "eth_dominance": global_metrics.get("eth_dominance", 0),
            "active_cryptocurrencies": global_metrics.get("active_cryptocurrencies", 0),
            "active_exchanges": global_metrics.get("active_exchanges", 0),
        }

    def generate_insights(self, symbol: str) -> list[str]:
        """특정 코인에 대한 인사이트 생성"""
        quote = self.cmc.get_quote(symbol)
        if not quote:
            return [f"❌ {symbol} 데이터를 찾을 수 없습니다."]

        insights = []

        # 가격 변동 인사이트
        if quote.percent_change_24h > 10:
            insights.append(
                f"🚀 {symbol}이 24시간 동안 {quote.percent_change_24h:.1f}% 급등했습니다."
            )
        elif quote.percent_change_24h < -10:
            insights.append(
                f"📉 {symbol}이 24시간 동안 {quote.percent_change_24h:.1f}% 급락했습니다."
            )

        # 7일 추세
        if quote.percent_change_7d > 0 and quote.percent_change_24h > 0:
            insights.append(f"📈 {symbol}은 지속적인 상승 추세를 보이고 있습니다.")
        elif quote.percent_change_7d < 0 and quote.percent_change_24h < 0:
            insights.append(f"📉 {symbol}은 지속적인 하락 추세를 보이고 있습니다.")
        elif quote.percent_change_7d * quote.percent_change_24h < 0:
            insights.append(
                f"🔄 {symbol}의 단기/중기 추세가 엇갈리고 있습니다. 변동성에 주의하세요."
            )

        # 볼륨 분석
        vol_to_cap = quote.volume_24h / quote.market_cap if quote.market_cap > 0 else 0
        if vol_to_cap > 0.3:
            insights.append(
                f"💰 거래량이 시가총액 대비 높습니다 ({vol_to_cap:.1%}). "
                "활발한 거래가 진행 중입니다."
            )

        # 기본 정보
        insights.append(
            f"💵 현재가: ${quote.price:,.2f} | "
            f"24h: {quote.percent_change_24h:+.1f}% | "
            f"7d: {quote.percent_change_7d:+.1f}%"
        )

        return insights if insights else [f"ℹ️ {symbol}에 대한 특별한 인사이트가 없습니다."]
