"""FastAPI 서버"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 전역 서비스 인스턴스
_service = None


def get_service():
    """서비스 인스턴스 반환 (lazy initialization)"""
    global _service
    if _service is None:
        from crypto_ai import CryptoAIService

        load_dotenv()
        api_key = os.getenv("CMC_API_KEY")
        if not api_key:
            raise RuntimeError("CMC_API_KEY 환경변수가 필요합니다.")
        _service = CryptoAIService(api_key)
    return _service


# ============================================================
# Pydantic Models
# ============================================================


class PriceResponse(BaseModel):
    symbol: str
    name: str
    price_usd: str
    change_1h: str
    change_24h: str
    change_7d: str
    volume_24h: str
    market_cap: str


class ChartAnalysisRequest(BaseModel):
    prices: list[float] = Field(..., min_length=20, description="가격 시계열")
    volumes: list[float] = Field(..., min_length=20, description="거래량 시계열")


class ChartAnalysisResponse(BaseModel):
    trend: str
    confidence: float
    probabilities: dict[str, float]
    indicators: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    device: str
    pytorch_version: str


class PredictionResponse(BaseModel):
    symbol: str
    model: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    volatility: float | None = None
    volume_change: float | None = None
    indicators: dict[str, float]
    market_sentiment: dict[str, float]  # Fear & Greed, BTC Dominance
    current_price: float
    price_change_24h: float


class AIInsightResponse(BaseModel):
    symbol: str
    predictions: dict  # interval -> prediction details
    sentiment: dict | None = None  # 센티멘트 데이터
    insight: str
    available_timeframes: list[str]


# ============================================================
# Lifespan
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 관리"""
    print("🚀 Crypto AI API 서버 시작")
    yield
    print("👋 서버 종료")


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Crypto AI Analysis API",
    description="MacBook MPS + PyTorch 기반 암호화폐 분석 서비스",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Endpoints
# ============================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """서비스 상태 확인"""
    import torch

    from crypto_ai.analyzer import get_device

    device = get_device()
    return HealthResponse(
        status="healthy",
        device=str(device),
        pytorch_version=torch.__version__,
    )


@app.get("/price/{symbol}", response_model=PriceResponse, tags=["Price"])
async def get_price(symbol: str):
    """특정 코인 시세 조회"""
    service = get_service()
    result = service.get_price(symbol)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return PriceResponse(**result)


@app.get("/prices", tags=["Price"])
async def get_multiple_prices(
    symbols: Annotated[str, Query(description="쉼표 구분 심볼 (예: BTC,ETH,AVAX)")]
):
    """여러 코인 시세 동시 조회"""
    service = get_service()
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    results = []
    for symbol in symbol_list:
        result = service.get_price(symbol)
        results.append(result)

    return {"quotes": results}


@app.post("/analyze/chart", response_model=ChartAnalysisResponse, tags=["Analysis"])
async def analyze_chart(request: ChartAnalysisRequest):
    """차트 기술적 분석"""
    if len(request.prices) != len(request.volumes):
        raise HTTPException(400, "prices와 volumes 길이가 일치해야 합니다.")

    service = get_service()
    result = service.analyze_chart(request.prices, request.volumes)

    return ChartAnalysisResponse(
        trend=result["trend"].value,
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        indicators=result["indicators"],
    )


@app.get("/insights/market", tags=["Insights"])
async def get_market_insights(
    limit: Annotated[int, Query(ge=10, le=200)] = 50,
):
    """전체 시장 인사이트"""
    service = get_service()
    return service.get_market_insights(limit=limit)


@app.get("/insights/{symbol}", tags=["Insights"])
async def get_coin_insights(symbol: str):
    """특정 코인 인사이트"""
    service = get_service()
    insights = service.get_coin_insights(symbol)

    return {"symbol": symbol.upper(), "insights": insights}


@app.get("/insights/ai/{symbol}", response_model=AIInsightResponse, tags=["AI Insights"])
async def get_ai_insight(symbol: str):
    """
    LLM 기반 멀티 타임프레임 AI 인사이트

    - **symbol**: 코인 심볼 (예: BTC, AVAX)

    모든 학습된 타임프레임(1h, 4h, 1d)의 예측을 수집하고,
    GPT-4o-mini를 통해 종합적인 투자 인사이트를 생성합니다.

    Returns:
    - 타임프레임별 예측 결과
    - LLM 생성 종합 인사이트
    """
    from crypto_ai.llm_insight import generate_ai_insight

    try:
        result = generate_ai_insight(symbol.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"인사이트 생성 실패: {str(e)}")

    if not result["predictions"]:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} 학습된 모델이 없습니다. 먼저 모델을 학습하세요."
        )

    return AIInsightResponse(**result)


@app.get("/insights/ai/{symbol}/stream", tags=["AI Insights"])
async def get_ai_insight_stream(symbol: str):
    """
    LLM 기반 멀티 타임프레임 AI 인사이트 (SSE 스트리밍)

    - **symbol**: 코인 심볼 (예: BTC, AVAX)

    Server-Sent Events (SSE)를 통해 분석 진행 상황을 실시간으로 전송합니다.

    Progress Events:
    - `progress`: 진행 상황 업데이트
      - `step`: 현재 단계 (init, load_model, fetch_data, predict, collect_sentiment, generate_insight, complete)
      - `status`: 한글 상태 메시지
      - `progress`: 0.0 ~ 1.0 진행률
      - `details`: 추가 정보

    - `result`: 최종 결과 (AIInsightResponse 형식)

    - `error`: 오류 발생 시

    Example (JavaScript):
    ```javascript
    const eventSource = new EventSource('/insights/ai/BTC/stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'progress') {
            console.log(`${data.status} (${Math.round(data.progress * 100)}%)`);
        } else if (data.type === 'result') {
            console.log('Result:', data.data);
            eventSource.close();
        }
    };
    ```
    """
    from crypto_ai.llm_insight import generate_ai_insight, ProgressSteps

    async def event_generator():
        progress_events = []

        def progress_callback(step: str, status: str, progress: float, details: dict | None):
            """진행 상황 콜백 - 이벤트 큐에 추가"""
            progress_events.append({
                "type": "progress",
                "step": step,
                "status": status,
                "progress": progress,
                "details": details or {},
            })

        # 별도 스레드에서 실행
        import concurrent.futures

        def run_insight():
            return generate_ai_insight(
                symbol.upper(),
                progress_callback=progress_callback,
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_insight)

            # 진행 상황 이벤트 전송
            last_sent = 0
            while not future.done():
                await asyncio.sleep(0.1)

                # 새 이벤트 전송
                while last_sent < len(progress_events):
                    event = progress_events[last_sent]
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    last_sent += 1

            # 남은 이벤트 전송
            while last_sent < len(progress_events):
                event = progress_events[last_sent]
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                last_sent += 1

            # 결과 가져오기
            try:
                result = future.result()
                if not result["predictions"]:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'{symbol.upper()} 학습된 모델이 없습니다.'}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'result', 'data': result}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        },
    )


@app.get("/predict/{symbol}", response_model=PredictionResponse, tags=["AI Prediction"])
async def predict_price(
    symbol: str,
    model: Annotated[str, Query(description="모델 타입")] = "transformer",
    interval: Annotated[str, Query(description="타임프레임 (1h, 4h, 1d)")] = "1h",
):
    """
    AI 모델로 가격 방향 예측

    - **symbol**: 코인 심볼 (예: BTC, ETH)
    - **model**: 모델 타입 (transformer 또는 lstm)
    - **interval**: 타임프레임 (1h: 1시간봉, 4h: 4시간봉, 1d: 일봉)

    Returns:
    - 가격 방향 예측 (상승/하락/횡보)
    - 확률 분포
    - 변동성, 거래량 예측 (transformer + multi-task)
    - 현재 기술적 지표
    """
    from pathlib import Path

    import torch

    from crypto_ai.preprocessing import DataPipeline, DataConfig, FEATURE_COLUMNS, INPUT_SIZE
    from crypto_ai.analyzer import get_device

    symbol = symbol.upper()

    # interval 유효성 검사
    if interval not in ["1h", "4h", "1d"]:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 interval: {interval}. 1h, 4h, 1d 중 선택하세요.")

    # 체크포인트 경로 (코인별/타임프레임별 디렉토리)
    if model == "transformer":
        checkpoint_path = Path("checkpoints/transformer") / symbol / interval / "best.pt"
    else:
        checkpoint_path = Path("checkpoints/lstm") / symbol / interval / "best.pt"

    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} ({interval}) 체크포인트가 없습니다. 먼저 모델을 학습하세요: scripts/train_transformer.py --symbol {symbol} --interval {interval}"
        )

    device = get_device()

    # 데이터 수집 (interval별 적절한 설정)
    days_map = {"1h": 7, "4h": 30, "1d": 90}
    seq_len_map = {"1h": 60, "4h": 60, "1d": 30}
    days = days_map.get(interval, 7)
    seq_len = seq_len_map.get(interval, 60)
    config = DataConfig(symbol=symbol.upper(), interval=interval, days=days, sequence_length=seq_len)
    pipeline = DataPipeline(config)

    try:
        df = pipeline.fetch_data()
        df = pipeline.compute_features(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 수집 실패: {str(e)}")

    # 특성 준비 (13개 특성)
    features = pipeline.normalize_features(df, FEATURE_COLUMNS, fit=True)

    seq_len = config.sequence_length
    if len(features) < seq_len:
        raise HTTPException(status_code=400, detail=f"데이터 부족: {len(features)}/{seq_len}")

    x = torch.tensor(features[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)

    # 예측
    direction_names = ["하락", "횡보", "상승"]
    volatility_val = None
    volume_val = None

    if model == "transformer":
        from crypto_ai.transformer import CryptoTransformer

        checkpoint = torch.load(checkpoint_path, map_location=device)
        multi_task = checkpoint.get("multi_task", False)

        transformer = CryptoTransformer(
            input_size=INPUT_SIZE, d_model=64, num_heads=4, num_layers=3, multi_task=multi_task
        )
        transformer.load_state_dict(checkpoint["model_state_dict"])
        transformer = transformer.to(device)
        transformer.eval()

        with torch.no_grad():
            outputs = transformer(x)
            probs = torch.softmax(outputs["direction"], dim=-1).cpu().numpy()[0]

            if multi_task:
                volatility_val = float(outputs["volatility"].cpu().numpy()[0][0])
                volume_val = float(outputs["volume"].cpu().numpy()[0][0])
    else:
        from crypto_ai.analyzer import ChartAnalyzer

        checkpoint = torch.load(checkpoint_path, map_location=device)
        lstm = ChartAnalyzer(input_size=INPUT_SIZE, hidden_size=64, num_layers=2)
        lstm.load_state_dict(checkpoint["model_state_dict"])
        lstm = lstm.to(device)
        lstm.eval()

        with torch.no_grad():
            probs = lstm(x).cpu().numpy()[0]

    direction_idx = int(probs.argmax())
    latest = df.iloc[-1]

    # 24시간 전 대비 변동률 계산
    if len(df) >= 24:
        price_24h_ago = df.iloc[-24]['close']
        change_24h = (latest['close'] - price_24h_ago) / price_24h_ago * 100
    else:
        change_24h = latest['returns'] * 100

    return PredictionResponse(
        symbol=symbol,
        model=model,
        prediction=direction_names[direction_idx],
        confidence=float(probs[direction_idx]),
        probabilities={
            "하락": float(probs[0]),
            "횡보": float(probs[1]),
            "상승": float(probs[2]),
        },
        volatility=volatility_val,
        volume_change=volume_val,
        indicators={
            "rsi": float(latest["rsi"]),
            "macd": float(latest["macd"]),
            "bb_position": float(latest["bb_position"]),
        },
        market_sentiment={
            "fear_greed": float(latest["fear_greed"]),
            "btc_dominance": float(latest["btc_dominance"]),
        },
        current_price=float(latest["close"]),
        price_change_24h=float(change_24h),
    )
