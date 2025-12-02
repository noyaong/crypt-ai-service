"""Pydantic 모델 및 데이터 클래스"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TrendDirection(Enum):
    """추세 방향"""
    BULLISH = "상승"
    BEARISH = "하락"
    NEUTRAL = "횡보"


@dataclass
class CryptoQuote:
    """암호화폐 시세 정보"""
    symbol: str
    name: str
    price: float
    volume_24h: float
    market_cap: float
    percent_change_1h: float
    percent_change_24h: float
    percent_change_7d: float
    last_updated: datetime


@dataclass
class ChartAnalysisResult:
    """차트 분석 결과"""
    trend: TrendDirection
    confidence: float
    probabilities: dict[str, float]
    indicators: dict[str, float]


@dataclass
class MarketSentiment:
    """시장 센티먼트"""
    positive_ratio: float
    avg_change_24h: float
    avg_change_7d: float
    fear_greed_index: float
    sentiment: str
