"""FastAPI 서버"""

import os
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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


@app.get("/predict/{symbol}", response_model=PredictionResponse, tags=["AI Prediction"])
async def predict_price(
    symbol: str,
    model: Annotated[str, Query(description="모델 타입")] = "transformer",
):
    """
    AI 모델로 가격 방향 예측

    - **symbol**: 코인 심볼 (예: BTC, ETH)
    - **model**: 모델 타입 (transformer 또는 lstm)

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

    # 체크포인트 경로 (코인별 디렉토리)
    if model == "transformer":
        checkpoint_path = Path("checkpoints/transformer") / symbol / "best.pt"
    else:
        checkpoint_path = Path("checkpoints/lstm") / symbol / "best.pt"

    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} 체크포인트가 없습니다. 먼저 모델을 학습하세요: scripts/train_transformer.py --symbol {symbol}"
        )

    device = get_device()

    # 데이터 수집
    config = DataConfig(symbol=symbol.upper(), interval="1h", days=7, sequence_length=60)
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
        price_change_24h=float(latest["returns"] * 100),
    )
