"""LLM 기반 멀티 타임프레임 인사이트 생성 (센티멘트 포함)"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from dotenv import load_dotenv
from openai import OpenAI

from crypto_ai.analyzer import get_device
from crypto_ai.preprocessing import DataConfig, DataPipeline, FEATURE_COLUMNS, INPUT_SIZE
from crypto_ai.data_sources import SentimentAggregator


# 진행 상황 콜백 타입 정의
# callback(step: str, status: str, progress: float, details: dict | None)
ProgressCallback = Callable[[str, str, float, dict | None], None]


class ProgressSteps:
    """진행 상황 단계 정의"""
    INIT = "init"
    LOAD_MODEL = "load_model"
    FETCH_DATA = "fetch_data"
    PREDICT = "predict"
    COLLECT_SENTIMENT = "collect_sentiment"
    GENERATE_INSIGHT = "generate_insight"
    COMPLETE = "complete"

    LABELS = {
        INIT: "분석 초기화 중...",
        LOAD_MODEL: "예측 모델 로드 중...",
        FETCH_DATA: "시장 데이터 수집 중...",
        PREDICT: "AI 예측 생성 중...",
        COLLECT_SENTIMENT: "센티멘트 데이터 수집 중...",
        GENERATE_INSIGHT: "LLM 인사이트 생성 중...",
        COMPLETE: "분석 완료",
    }


@dataclass
class PredictionResult:
    """단일 타임프레임 예측 결과"""
    interval: str
    direction: str
    confidence: float
    probabilities: dict[str, float]
    volatility: float | None
    volume_change: float | None
    rsi: float
    macd: float
    bb_position: float
    fear_greed: float
    btc_dominance: float
    current_price: float
    price_change_24h: float


@dataclass
class SentimentData:
    """센티멘트 데이터"""
    fear_greed_value: int | None = None
    fear_greed_label: str | None = None
    news_total_posts: int = 0
    news_sentiment_score: float | None = None
    news_bullish_ratio: float | None = None
    news_bearish_ratio: float | None = None
    recent_headlines: list[dict] = field(default_factory=list)
    social_score: float | None = None
    twitter_followers: int | None = None
    reddit_subscribers: int | None = None
    overall_sentiment: str = "neutral"
    overall_score: float = 0.0


@dataclass
class MultiTimeframePrediction:
    """멀티 타임프레임 예측 결과"""
    symbol: str
    predictions: dict[str, PredictionResult]  # interval -> PredictionResult
    sentiment: SentimentData | None = None  # 센티멘트 데이터


class MultiTimeframePredictor:
    """멀티 타임프레임 예측기"""

    # 타임프레임별 설정
    TIMEFRAME_CONFIG = {
        "1h": {"days": 7, "seq_len": 60, "label": "1시간봉"},
        "4h": {"days": 30, "seq_len": 60, "label": "4시간봉"},
        "1d": {"days": 90, "seq_len": 30, "label": "일봉"},
    }

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        progress_callback: ProgressCallback | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = get_device()
        self.progress_callback = progress_callback

    def _emit_progress(
        self,
        step: str,
        progress: float,
        details: dict | None = None,
    ) -> None:
        """진행 상황 콜백 호출"""
        if self.progress_callback:
            status = ProgressSteps.LABELS.get(step, step)
            self.progress_callback(step, status, progress, details)

    def predict_single(self, symbol: str, interval: str) -> PredictionResult | None:
        """단일 타임프레임 예측"""
        symbol = symbol.upper()
        config = self.TIMEFRAME_CONFIG.get(interval)
        if not config:
            return None

        # 체크포인트 확인
        checkpoint_path = self.checkpoint_dir / "transformer" / symbol / interval / "best.pt"
        if not checkpoint_path.exists():
            return None

        # 데이터 수집
        data_config = DataConfig(
            symbol=symbol,
            interval=interval,
            days=config["days"],
            sequence_length=config["seq_len"]
        )
        pipeline = DataPipeline(data_config)

        try:
            df = pipeline.fetch_data()
            df = pipeline.compute_features(df)
        except Exception:
            return None

        # 특성 준비
        features = pipeline.normalize_features(df, FEATURE_COLUMNS, fit=True)
        seq_len = config["seq_len"]

        if len(features) < seq_len:
            return None

        x = torch.tensor(features[-seq_len:], dtype=torch.float32).unsqueeze(0).to(self.device)

        # 모델 로드 및 예측
        from crypto_ai.transformer import CryptoTransformer

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        multi_task = checkpoint.get("multi_task", False)

        model = CryptoTransformer(
            input_size=INPUT_SIZE,
            d_model=64,
            num_heads=4,
            num_layers=3,
            multi_task=multi_task,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        direction_names = ["하락", "횡보", "상승"]
        volatility_val = None
        volume_val = None

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs["direction"], dim=-1).cpu().numpy()[0]

            if multi_task:
                volatility_val = float(outputs["volatility"].cpu().numpy()[0][0])
                volume_val = float(outputs["volume"].cpu().numpy()[0][0])

        direction_idx = int(probs.argmax())
        latest = df.iloc[-1]

        # 24시간 변동률
        if interval == "1h" and len(df) >= 24:
            price_24h_ago = df.iloc[-24]["close"]
            change_24h = (latest["close"] - price_24h_ago) / price_24h_ago * 100
        elif interval == "4h" and len(df) >= 6:
            price_24h_ago = df.iloc[-6]["close"]
            change_24h = (latest["close"] - price_24h_ago) / price_24h_ago * 100
        else:
            change_24h = latest["returns"] * 100

        return PredictionResult(
            interval=interval,
            direction=direction_names[direction_idx],
            confidence=float(probs[direction_idx]),
            probabilities={
                "하락": float(probs[0]),
                "횡보": float(probs[1]),
                "상승": float(probs[2]),
            },
            volatility=volatility_val,
            volume_change=volume_val,
            rsi=float(latest["rsi"]),
            macd=float(latest["macd"]),
            bb_position=float(latest["bb_position"]),
            fear_greed=float(latest["fear_greed"]),
            btc_dominance=float(latest["btc_dominance"]),
            current_price=float(latest["close"]),
            price_change_24h=float(change_24h),
        )

    def predict_all(
        self,
        symbol: str,
        intervals: list[str] | None = None,
        include_sentiment: bool = True,
    ) -> MultiTimeframePrediction:
        """모든 타임프레임 예측 + 센티멘트 데이터"""
        intervals = intervals or ["1h", "4h", "1d"]
        predictions = {}

        # 진행상황: 예측 시작
        total_steps = len(intervals) + (1 if include_sentiment else 0)
        completed = 0

        for i, interval in enumerate(intervals):
            # 진행상황: 데이터 수집 중
            self._emit_progress(
                ProgressSteps.FETCH_DATA,
                progress=(completed / total_steps) * 0.6,
                details={"interval": interval, "label": self.TIMEFRAME_CONFIG[interval]["label"]},
            )

            # 진행상황: 예측 생성 중
            self._emit_progress(
                ProgressSteps.PREDICT,
                progress=((completed + 0.5) / total_steps) * 0.6,
                details={"interval": interval, "current": i + 1, "total": len(intervals)},
            )

            result = self.predict_single(symbol, interval)
            if result:
                predictions[interval] = result

            completed += 1

        # 센티멘트 데이터 수집
        sentiment_data = None
        if include_sentiment:
            self._emit_progress(
                ProgressSteps.COLLECT_SENTIMENT,
                progress=0.6,
                details={"symbol": symbol},
            )
            sentiment_data = self._collect_sentiment(symbol)

        return MultiTimeframePrediction(
            symbol=symbol.upper(),
            predictions=predictions,
            sentiment=sentiment_data,
        )

    def _collect_sentiment(self, symbol: str) -> SentimentData | None:
        """센티멘트 데이터 수집"""
        try:
            aggregator = SentimentAggregator()
            data = aggregator.get_comprehensive_sentiment(symbol)

            sentiment = SentimentData(
                overall_sentiment=data.get("overall_sentiment", "neutral"),
                overall_score=data.get("sentiment_score", 0.0),
            )

            # Fear & Greed
            if data.get("fear_greed"):
                fng = data["fear_greed"]
                sentiment.fear_greed_value = fng.get("value")
                sentiment.fear_greed_label = fng.get("value_classification")

            # 뉴스 센티멘트
            if data.get("news_sentiment"):
                news = data["news_sentiment"]
                sentiment.news_total_posts = news.get("total_posts", 0)
                sentiment.news_sentiment_score = news.get("sentiment_score")
                sentiment.news_bullish_ratio = news.get("bullish_ratio")
                sentiment.news_bearish_ratio = news.get("bearish_ratio")
                sentiment.recent_headlines = news.get("recent_headlines", [])[:5]

            # 소셜 지표
            if data.get("social_metrics"):
                social = data["social_metrics"]
                sentiment.social_score = social.get("social_score")
                sentiment.twitter_followers = social.get("twitter_followers")
                sentiment.reddit_subscribers = social.get("reddit_subscribers")

            return sentiment
        except Exception:
            return None


class LLMInsightGenerator:
    """OpenAI GPT를 활용한 인사이트 생성"""

    SYSTEM_PROMPT = """당신은 암호화폐 기술적 분석 및 센티멘트 분석 전문가입니다.
AI 모델의 멀티 타임프레임 예측 결과와 시장 센티멘트 데이터를 종합 분석하여 투자자에게 명확하고 실용적인 인사이트를 제공합니다.

분석 시 고려사항:
1. 타임프레임 간 컨플루언스 (일치/불일치)
2. 기술적 지표의 맥락 (RSI 과매수/과매도, MACD 방향 등)
3. 시장 심리 (Fear & Greed Index)
4. **뉴스 헤드라인 직접 분석**: 제공된 뉴스 헤드라인을 읽고, 각 뉴스가 해당 코인에 긍정적/부정적/중립적인지 직접 판단하세요. ETF 승인, 파트너십, 기술 업데이트 등은 긍정적, 해킹, 규제, 소송 등은 부정적입니다.
5. 소셜 미디어 활동 (커뮤니티 규모, 활성도)
6. 기술적 분석과 센티멘트 간의 괴리 또는 일치
7. 리스크 요인

응답 형식:
- 간결하고 명확하게
- 불릿 포인트 활용
- 한국어로 작성
- 기술적 분석과 센티멘트 분석을 구분하여 설명
- **뉴스 분석 섹션**: 주요 뉴스 2-3개를 언급하며 왜 긍정적/부정적인지 간략히 설명
- 투자 권유가 아닌 분석 관점으로"""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def _build_prompt(self, prediction: MultiTimeframePrediction) -> str:
        """예측 결과를 프롬프트로 변환"""
        symbol = prediction.symbol
        lines = [f"## {symbol} 멀티 타임프레임 AI 예측 결과\n"]

        timeframe_labels = {"1h": "단기 (1시간봉)", "4h": "중기 (4시간봉)", "1d": "장기 (일봉)"}

        for interval, result in prediction.predictions.items():
            label = timeframe_labels.get(interval, interval)
            lines.append(f"### {label}")
            lines.append(f"- 예측: {result.direction} (신뢰도: {result.confidence*100:.1f}%)")
            lines.append(f"- 확률: 하락 {result.probabilities['하락']*100:.1f}% / 횡보 {result.probabilities['횡보']*100:.1f}% / 상승 {result.probabilities['상승']*100:.1f}%")

            if result.volatility is not None:
                vol_label = "높음" if result.volatility > 0.5 else "보통" if result.volatility > -0.5 else "낮음"
                lines.append(f"- 예상 변동성: {vol_label} ({result.volatility:+.2f})")

            if result.volume_change is not None:
                vol_label = "증가" if result.volume_change > 0.3 else "감소" if result.volume_change < -0.3 else "유지"
                lines.append(f"- 예상 거래량: {vol_label} ({result.volume_change:+.2f})")

            lines.append(f"- RSI: {result.rsi:.1f}" + (" (과매수)" if result.rsi > 70 else " (과매도)" if result.rsi < 30 else ""))
            lines.append(f"- MACD: {result.macd:.4f}" + (" (양)" if result.macd > 0 else " (음)"))
            lines.append(f"- 볼린저밴드 위치: {result.bb_position*100:.1f}%")
            lines.append("")

        # 공통 정보 (첫 번째 예측에서 추출)
        if prediction.predictions:
            first = next(iter(prediction.predictions.values()))
            lines.append("### 시장 상황")
            fng = first.fear_greed
            fng_label = "극단적 공포" if fng < 25 else "공포" if fng < 45 else "중립" if fng < 55 else "탐욕" if fng < 75 else "극단적 탐욕"
            lines.append(f"- Fear & Greed Index: {fng:.0f} ({fng_label})")
            lines.append(f"- BTC 도미넌스: {first.btc_dominance:.1f}%")
            lines.append(f"- 현재가: ${first.current_price:,.2f}")
            lines.append(f"- 24h 변동: {first.price_change_24h:+.2f}%")
            lines.append("")

        # 센티멘트 데이터 추가
        if prediction.sentiment:
            sent = prediction.sentiment
            lines.append("### 센티멘트 분석")

            # Fear & Greed (센티멘트 수집기에서 가져온 값)
            if sent.fear_greed_value is not None:
                lines.append(f"- Fear & Greed Index (실시간): {sent.fear_greed_value} ({sent.fear_greed_label})")

            # 뉴스 센티멘트
            if sent.news_sentiment_score is not None:
                news_label = "긍정적" if sent.news_sentiment_score > 0.2 else "부정적" if sent.news_sentiment_score < -0.2 else "중립"
                lines.append(f"- 뉴스 센티멘트: {news_label} (점수: {sent.news_sentiment_score:+.2f})")
                if sent.news_bullish_ratio is not None:
                    lines.append(f"  - 긍정 뉴스 비율: {sent.news_bullish_ratio*100:.1f}%")
                    lines.append(f"  - 부정 뉴스 비율: {sent.news_bearish_ratio*100:.1f}%")

            # 최근 헤드라인 (GPT가 직접 분석할 수 있도록 전체 제목 전달)
            if sent.recent_headlines:
                lines.append("- 최근 뉴스 헤드라인 (아래 내용을 직접 분석하여 센티멘트를 판단해주세요):")
                for i, headline in enumerate(sent.recent_headlines[:5], 1):
                    title = headline.get('title', '')
                    published = headline.get('published_at', '')[:10]  # YYYY-MM-DD
                    lines.append(f"  {i}. [{published}] {title}")

            # 소셜 지표
            if sent.social_score is not None:
                social_label = "높음" if sent.social_score > 70 else "보통" if sent.social_score > 40 else "낮음"
                lines.append(f"- 소셜 활성도: {social_label} (점수: {sent.social_score:.1f}/100)")
                if sent.twitter_followers:
                    lines.append(f"  - Twitter 팔로워: {sent.twitter_followers:,}")
                if sent.reddit_subscribers:
                    lines.append(f"  - Reddit 구독자: {sent.reddit_subscribers:,}")

            # 종합 센티멘트
            overall_emoji = "🟢" if sent.overall_sentiment == "bullish" else "🔴" if sent.overall_sentiment == "bearish" else "🟡"
            lines.append(f"- 종합 센티멘트: {overall_emoji} {sent.overall_sentiment.upper()} (점수: {sent.overall_score:+.3f})")

        return "\n".join(lines)

    def generate_insight(self, prediction: MultiTimeframePrediction) -> str:
        """LLM을 통한 인사이트 생성"""
        if not prediction.predictions:
            return "예측 데이터가 없습니다. 먼저 모델을 학습하세요."

        user_prompt = self._build_prompt(prediction)
        user_prompt += "\n\n위 데이터를 기반으로 종합적인 투자 인사이트를 분석해주세요."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        return response.choices[0].message.content or "인사이트 생성 실패"


def generate_ai_insight(
    symbol: str,
    checkpoint_dir: str = "checkpoints",
    include_sentiment: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """
    멀티 타임프레임 예측 + 센티멘트 + LLM 인사이트 생성

    Args:
        symbol: 코인 심볼 (예: BTC, AVAX)
        checkpoint_dir: 체크포인트 디렉토리
        include_sentiment: 센티멘트 데이터 포함 여부
        progress_callback: 진행상황 콜백 함수
            - callback(step: str, status: str, progress: float, details: dict | None)
            - step: ProgressSteps 상수 (init, load_model, fetch_data, predict, collect_sentiment, generate_insight, complete)
            - status: 사용자에게 표시할 한글 상태 메시지
            - progress: 0.0 ~ 1.0 진행률
            - details: 추가 정보 (interval, current, total 등)

    Returns:
        {
            "symbol": str,
            "predictions": dict,
            "sentiment": dict | None,
            "insight": str,
            "available_timeframes": list,
        }
    """
    def emit(step: str, progress: float, details: dict | None = None) -> None:
        if progress_callback:
            status = ProgressSteps.LABELS.get(step, step)
            progress_callback(step, status, progress, details)

    # 진행상황: 초기화
    emit(ProgressSteps.INIT, 0.0, {"symbol": symbol})

    # 진행상황: 모델 로드 시작
    emit(ProgressSteps.LOAD_MODEL, 0.05, {"checkpoint_dir": checkpoint_dir})

    # 1. 멀티 타임프레임 예측 + 센티멘트
    predictor = MultiTimeframePredictor(checkpoint_dir, progress_callback=progress_callback)
    multi_pred = predictor.predict_all(symbol, include_sentiment=include_sentiment)

    # 진행상황: LLM 인사이트 생성 시작
    emit(ProgressSteps.GENERATE_INSIGHT, 0.7, {"model": "gpt-4o-mini"})

    # 2. LLM 인사이트 생성
    try:
        generator = LLMInsightGenerator()
        insight = generator.generate_insight(multi_pred)
    except ValueError as e:
        insight = f"LLM 인사이트 생성 불가: {e}"

    # 진행상황: 완료
    emit(ProgressSteps.COMPLETE, 1.0, {"symbol": symbol})

    # 3. 결과 반환
    predictions_dict = {}
    for interval, pred in multi_pred.predictions.items():
        predictions_dict[interval] = {
            "direction": pred.direction,
            "confidence": pred.confidence,
            "probabilities": pred.probabilities,
            "volatility": pred.volatility,
            "volume_change": pred.volume_change,
            "indicators": {
                "rsi": pred.rsi,
                "macd": pred.macd,
                "bb_position": pred.bb_position,
            },
            "market": {
                "fear_greed": pred.fear_greed,
                "btc_dominance": pred.btc_dominance,
            },
            "price": {
                "current": pred.current_price,
                "change_24h": pred.price_change_24h,
            },
        }

    # 센티멘트 데이터 변환
    sentiment_dict = None
    if multi_pred.sentiment:
        sent = multi_pred.sentiment
        sentiment_dict = {
            "overall_sentiment": sent.overall_sentiment,
            "overall_score": sent.overall_score,
            "fear_greed": {
                "value": sent.fear_greed_value,
                "label": sent.fear_greed_label,
            } if sent.fear_greed_value is not None else None,
            "news": {
                "total_posts": sent.news_total_posts,
                "sentiment_score": sent.news_sentiment_score,
                "bullish_ratio": sent.news_bullish_ratio,
                "bearish_ratio": sent.news_bearish_ratio,
                "recent_headlines": sent.recent_headlines,
            } if sent.news_total_posts > 0 else None,
            "social": {
                "score": sent.social_score,
                "twitter_followers": sent.twitter_followers,
                "reddit_subscribers": sent.reddit_subscribers,
            } if sent.social_score is not None else None,
        }

    return {
        "symbol": symbol.upper(),
        "predictions": predictions_dict,
        "sentiment": sentiment_dict,
        "insight": insight,
        "available_timeframes": list(multi_pred.predictions.keys()),
    }
