# Crypto AI Analysis Service

PyTorch 기반 암호화폐 AI 분석 서비스 (CUDA / MPS / CPU 자동 지원)

---

## 목차

- [개요](#개요)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [CLI 명령어](#cli-명령어)
- [API 서버](#api-서버)
- [모델 학습](#모델-학습)
- [Python SDK](#python-sdk)
- [프로젝트 구조](#프로젝트-구조)
- [데이터 소스](#데이터-소스)
- [개발](#개발)

---

## 개요

### 주요 기능

| 기능 | 설명 |
|------|------|
| **시세 조회** | CoinMarketCap, Binance, CoinGecko 연동 |
| **AI 예측** | Transformer/LSTM 모델로 가격 방향 예측 |
| **멀티태스크 학습** | 가격 방향 + 변동성 + 거래량 동시 예측 |
| **기술적 분석** | RSI, MACD, 볼린저밴드, ATR |
| **시장 인사이트** | Fear & Greed Index, 센티먼트 분석 |
| **MPS 가속** | Apple Silicon GPU 활용 |

### 모델 비교

| 항목 | LSTM | Transformer |
|------|------|-------------|
| 파라미터 | ~50K | ~157K |
| 멀티태스크 | ❌ | ✅ |
| Attention | ❌ | ✅ |
| 권장 용도 | 빠른 테스트 | 프로덕션 |

> 자세한 모델 아키텍처는 [docs/MODELS.md](docs/MODELS.md) 참조

---

## 설치

### 요구사항

- Python 3.11+
- uv 패키지 매니저
- GPU (선택, 자동 감지)

### 지원 디바이스

| 우선순위 | 디바이스 | 환경 |
|----------|----------|------|
| 1 | CUDA | NVIDIA GPU (Windows/Linux) |
| 2 | MPS | Apple Silicon (M1/M2/M3) |
| 3 | CPU | fallback (모든 환경) |

> 디바이스는 자동 감지됩니다. GPU가 없어도 CPU로 동작합니다.

### 설치 방법

```bash
# 1. uv 설치 (없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 프로젝트 클론 및 이동
cd crypto-ai-service

# 3. 의존성 설치
uv sync              # 기본
uv sync --all-extras # 전체 (API, 시각화, 개발도구)

# 4. 환경변수 설정 (선택)
cp .env.example .env
# CMC_API_KEY 입력 (CoinMarketCap 사용 시 필요)
```

---

## 빠른 시작

### 1. 시스템 확인

```bash
uv run crypto-ai check
```

### 2. AI 예측 (모델 학습 필요)

```bash
# 먼저 모델 학습 (최초 1회)
uv run python scripts/train_transformer.py --symbol BTC --days 90 --epochs 20 --multi-task

# 예측 실행
uv run crypto-ai predict BTC
```

### 3. 시세 조회

```bash
uv run crypto-ai price BTC ETH SOL
```

---

## CLI 명령어

### `check` - 시스템 상태

```bash
uv run crypto-ai check
```
MPS 가용성, PyTorch 버전, GPU 벤치마크 확인

### `price` - 시세 조회

```bash
uv run crypto-ai price BTC
uv run crypto-ai price ETH AVAX SOL  # 여러 코인
```
현재가, 1h/24h/7d 변동률, 거래량, 시가총액

### `market` - 시장 인사이트

```bash
uv run crypto-ai market
```
Fear & Greed Index, 센티먼트, Top 상승/하락 코인

### `insight` - 코인 인사이트

```bash
uv run crypto-ai insight AVAX
```
개별 코인에 대한 AI 생성 인사이트

### `predict` - AI 예측

```bash
uv run crypto-ai predict BTC                    # Transformer, 1시간봉 (기본)
uv run crypto-ai predict BTC --interval 4h      # 4시간봉
uv run crypto-ai predict BTC -i 1d              # 일봉
uv run crypto-ai predict ETH --model lstm       # LSTM
uv run crypto-ai predict SOL -m transformer     # 명시적 지정
```

**타임프레임 옵션:**

| 옵션 | 설명 | 특징 |
|------|------|------|
| `1h` | 1시간봉 (기본) | 단기 트레이딩, 빠른 신호 |
| `4h` | 4시간봉 | 중기 트레이딩, 노이즈 감소 |
| `1d` | 일봉 | 장기 투자, 큰 추세 파악 |

**출력 정보:**
- 가격 방향 예측 (상승/하락/횡보)
- 확률 분포 (시각화)
- 변동성/거래량 예측 (Transformer)
- 현재 기술적 지표 (RSI, MACD, 볼린저)

### `insight-ai` - LLM 멀티 타임프레임 분석

```bash
uv run crypto-ai insight-ai AVAX
```

**기능:**
- 모든 타임프레임(1h, 4h, 1d) 예측 자동 수집
- 뉴스/소셜 센티멘트 분석
- GPT-4o-mini를 통한 종합 인사이트 생성
- 타임프레임 간 컨플루언스(일치/불일치) 분석

**출력 예시:**
```
【단기 (1시간봉)】 예측: 하락 (신뢰도: 41.3%)
【중기 (4시간봉)】 예측: 상승 (신뢰도: 47.7%)
【장기 (일봉)】    예측: 횡보 (신뢰도: 35.4%)

📰 센티멘트 분석
   뉴스 센티멘트: 중립 (점수: +0.10)
   소셜 활성도: 보통 (점수: 45.2/100)
   종합: 🟡 NEUTRAL (점수: +0.050)

💡 AI 종합 분석: 기술적 분석과 센티멘트를 종합한 인사이트...
```

**센티멘트 데이터 소스:**
- Fear & Greed Index (Alternative.me)
- 뉴스 센티멘트 (CryptoPanic API - 선택적)
- 소셜 미디어 지표 (CoinGecko community/developer data)

> **필수**: `OPENAI_API_KEY` 환경변수 설정 필요
> **선택**: `CRYPTOPANIC_API_KEY` 환경변수 (뉴스 센티멘트 향상)

---

## API 서버

### 서버 실행

```bash
# 개발
uv run uvicorn crypto_ai.api:app --reload

# 프로덕션
uv run uvicorn crypto_ai.api:app --host 0.0.0.0 --port 8000
```

### 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서비스 상태 |
| GET | `/price/{symbol}` | 시세 조회 |
| GET | `/prices?symbols=BTC,ETH` | 다중 시세 |
| GET | `/predict/{symbol}` | AI 예측 (단일 타임프레임) |
| GET | `/insights/ai/{symbol}` | **LLM 멀티 타임프레임 분석** |
| GET | `/insights/ai/{symbol}/stream` | **LLM 분석 (SSE 스트리밍)** |
| GET | `/insights/market` | 시장 인사이트 |
| GET | `/insights/{symbol}` | 코인 인사이트 |
| POST | `/analyze/chart` | 차트 분석 |

### 예시

```bash
# AI 예측 (1시간봉, 기본)
curl http://localhost:8000/predict/BTC

# AI 예측 (4시간봉)
curl "http://localhost:8000/predict/BTC?interval=4h"

# AI 예측 (일봉, LSTM)
curl "http://localhost:8000/predict/BTC?interval=1d&model=lstm"

# 응답
{
  "symbol": "BTC",
  "model": "transformer",
  "prediction": "하락",
  "confidence": 0.443,
  "probabilities": {"하락": 0.443, "횡보": 0.229, "상승": 0.328},
  "volatility": 0.56,
  "volume_change": 0.01,
  "indicators": {"rsi": 46.6, "macd": -770.5, "bb_position": 0.64},
  "current_price": 86290.76,
  "price_change_24h": -0.33
}
```

### SSE 스트리밍 (실시간 진행상황)

프론트엔드에서 분석 진행상황을 실시간으로 표시하려면 SSE 스트리밍 엔드포인트를 사용하세요:

```javascript
// JavaScript 예시
const eventSource = new EventSource('/insights/ai/BTC/stream');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'progress') {
        // 진행상황 표시
        console.log(`${data.status} (${Math.round(data.progress * 100)}%)`);
        // 예: "예측 모델 로드 중... (5%)"
        // 예: "AI 예측 생성 중... (30%)"
        // 예: "LLM 인사이트 생성 중... (70%)"
        updateProgressBar(data.progress);
    } else if (data.type === 'result') {
        // 최종 결과
        console.log('분석 완료:', data.data);
        eventSource.close();
    } else if (data.type === 'error') {
        console.error('오류:', data.message);
        eventSource.close();
    }
};
```

**진행상황 단계:**

| Step | Status | Progress |
|------|--------|----------|
| `init` | 분석 초기화 중... | 0% |
| `load_model` | 예측 모델 로드 중... | 5% |
| `fetch_data` | 시장 데이터 수집 중... | 10-50% |
| `predict` | AI 예측 생성 중... | 20-60% |
| `collect_sentiment` | 센티멘트 데이터 수집 중... | 60% |
| `generate_insight` | LLM 인사이트 생성 중... | 70% |
| `complete` | 분석 완료 | 100% |

**API 문서**: http://localhost:8000/docs

---

## 모델 학습

### LSTM 학습

```bash
uv run python scripts/train.py \
    --symbol BTC \
    --days 180 \
    --epochs 50 \
    --batch-size 32
```

### Transformer 학습 (권장)

```bash
uv run python scripts/train_transformer.py \
    --symbol BTC \
    --interval 1h \
    --days 365 \
    --epochs 100 \
    --multi-task \
    --d-model 64 \
    --num-heads 4 \
    --num-layers 3
```

### 학습 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--symbol` | 코인 심볼 | BTC |
| `--interval` | 타임프레임 (1h, 4h, 1d) | 1h |
| `--days` | 학습 데이터 기간 | 365 |
| `--epochs` | 에폭 수 | 100 |
| `--batch-size` | 배치 크기 | 32 |
| `--lr` | 학습률 | 0.0001 |
| `--multi-task` | 멀티태스크 활성화 | False |
| `--d-model` | 모델 차원 | 64 |
| `--num-heads` | Attention 헤드 수 | 4 |
| `--num-layers` | 레이어 수 | 3 |

### 타임프레임별 학습 예시

```bash
# 1시간봉 (단기 트레이딩)
uv run python scripts/train_transformer.py --symbol BTC --interval 1h --days 90 --epochs 20 --multi-task

# 4시간봉 (중기 트레이딩)
uv run python scripts/train_transformer.py --symbol BTC --interval 4h --days 180 --epochs 30 --multi-task

# 일봉 (장기 투자) - days 730, seq-length 30 권장
uv run python scripts/train_transformer.py --symbol BTC --interval 1d --days 730 --seq-length 30 --epochs 50 --multi-task
```

| 타임프레임 | 권장 days | 권장 seq-length | 데이터 수 | 특징 |
|------------|-----------|-----------------|-----------|------|
| `1h` | 90~180 | 60 (기본) | 2,160~4,320 | 빠른 신호, 노이즈 많음 |
| `4h` | 180~365 | 60 (기본) | 1,080~2,190 | 균형 잡힌 신호 |
| `1d` | **730** | **30** | ~730 | 큰 추세, 충분한 데이터 필요 |

> **주의**: 일봉(`1d`)은 `--days 730 --seq-length 30` 권장

### 학습 옵션 예시

```bash
uv run python scripts/train_transformer.py --symbol AVAX --interval 1h --days 90 --epochs 20 --multi-task
```

| 옵션 | 의미 |
|------|------|
| `--interval 1h` | 1시간봉 데이터 사용 |
| `--days 90` | 90일치 = 약 2,160개 데이터 |
| `--epochs 20` | 전체 데이터를 20번 반복 학습 |
| `--multi-task` | 방향 + 변동성 + 거래량 동시 예측 |

### Epoch (에폭) 이란?

**Epoch**은 전체 학습 데이터를 한 번 완전히 학습하는 단위입니다.

```
예시: 8,760개 데이터 (1년치 1시간봉), batch_size=32

1 Epoch = 8,760 ÷ 32 = 274번의 학습 반복
10 Epochs = 전체 데이터를 10번 반복 학습
```

| Epoch 수 | 특징 |
|----------|------|
| 적음 (5~10) | 빠른 테스트, 과소적합 위험 |
| 보통 (20~50) | 일반적인 학습 |
| 많음 (100+) | 정밀 학습, 과적합 주의 |

> **Early Stopping**: 검증 손실이 개선되지 않으면 자동 종료됩니다 (기본 patience=15)

### 학습 모니터링

```bash
uv run tensorboard --logdir runs
# http://localhost:6006
```

### 체크포인트

체크포인트는 **코인별 + 타임프레임별**로 저장됩니다:

```
checkpoints/
├── transformer/
│   ├── BTC/
│   │   ├── 1h/best.pt      # BTC 1시간봉
│   │   ├── 4h/best.pt      # BTC 4시간봉
│   │   └── 1d/best.pt      # BTC 일봉
│   └── AVAX/
│       └── 1h/best.pt      # AVAX 1시간봉
└── lstm/
    └── BTC/
        └── 1h/best.pt
```

- Attention 시각화: `checkpoints/transformer/{SYMBOL}/{INTERVAL}/attention.png`

### 주기적 재학습 (배치 작업)

모델 성능 유지를 위해 **주기적으로 재학습**이 필요합니다.

| 시장 상황 | 권장 주기 |
|----------|----------|
| 일반 | 주 1회 |
| 변동성 큰 시장 | 일 1회 |
| 프로덕션 | 일 1회 + 성능 모니터링 |

**Crontab 예시 (매주 일요일 새벽 3시):**

```bash
# crontab -e
0 3 * * 0 cd /path/to/crypto-ai-service && uv run python scripts/train_transformer.py --symbol BTC --interval 1h --days 90 --epochs 20 --multi-task >> logs/train.log 2>&1
```

**여러 코인 + 타임프레임 학습:**

```bash
for symbol in BTC ETH AVAX SOL; do
  for interval in 1h 4h 1d; do
    uv run python scripts/train_transformer.py --symbol $symbol --interval $interval --days 90 --epochs 20 --multi-task
  done
done
```

> ⚠️ 체크포인트가 없으면 `predict` 명령이 실패합니다. 최초 1회는 반드시 수동 학습 필요.

---

## Python SDK

### 기본 사용

```python
from crypto_ai import CryptoAIService

service = CryptoAIService()

# 시세 조회
price = service.get_price("BTC")
print(price)

# 시장 인사이트
insights = service.get_market_insights()
print(f"Fear & Greed: {insights['sentiment_analysis']['fear_greed_index']}")
```

### 무료 데이터 수집 (API 키 불필요)

```python
from crypto_ai import BinanceClient, CoinGeckoClient, AlternativeMeClient

# Binance - OHLCV 데이터
binance = BinanceClient()
df = binance.get_klines("BTCUSDT", interval="1h", limit=1000)
df = binance.get_historical_klines("BTCUSDT", interval="1h", days=365)

# CoinGecko - 시세 + 메타데이터
coingecko = CoinGeckoClient()
markets = coingecko.get_coin_markets(per_page=10)

# Alternative.me - Fear & Greed Index
fng = AlternativeMeClient()
index = fng.get_fear_greed_index()
print(f"Fear & Greed: {index['value']} ({index['value_classification']})")
```

### Transformer 모델 직접 사용

```python
import torch
from crypto_ai.transformer import CryptoTransformer
from crypto_ai.preprocessing import DataPipeline, DataConfig

# 데이터 준비
config = DataConfig(symbol="BTC", interval="1h", days=7)
pipeline = DataPipeline(config)
df = pipeline.fetch_data()
df = pipeline.compute_features(df)

# 모델 로드
model = CryptoTransformer(input_size=9, d_model=64, num_heads=4, num_layers=3, multi_task=True)
checkpoint = torch.load("checkpoints/transformer/best.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 예측
x = torch.randn(1, 60, 9)  # (batch, seq_len, features)
result = model.predict(x)
print(f"예측: {result['direction_name']}, 신뢰도: {result['confidence']}")
```

---

## 프로젝트 구조

```
crypto-ai-service/
├── README.md                   # 이 파일
├── ROADMAP.md                  # 개발 로드맵
├── docs/
│   └── MODELS.md               # 모델 아키텍처 상세
├── src/crypto_ai/
│   ├── __init__.py             # CryptoAIService
│   ├── cli.py                  # CLI 명령어
│   ├── api.py                  # FastAPI 서버
│   ├── client.py               # CoinMarketCap 클라이언트
│   ├── data_sources.py         # Binance/CoinGecko/Alternative.me/CryptoPanic
│   ├── analyzer.py             # LSTM 모델
│   ├── transformer.py          # Transformer 모델
│   ├── preprocessing.py        # 데이터 전처리
│   ├── insight.py              # 인사이트 생성
│   ├── llm_insight.py          # LLM 기반 멀티타임프레임 분석
│   └── models.py               # Pydantic 모델
├── scripts/
│   ├── train.py                # LSTM 학습
│   └── train_transformer.py    # Transformer 학습
├── checkpoints/                # 모델 체크포인트
├── runs/                       # TensorBoard 로그
└── tests/                      # 테스트
```

---

## 데이터 소스

| 소스 | 용도 | API 키 | Rate Limit | 문서 |
|------|------|--------|------------|------|
| **Binance** | OHLCV 히스토리 | 불필요 | 1200/분 | [링크](https://binance-docs.github.io/apidocs/spot/en/) |
| **CoinGecko** | 시세, 메타데이터, 소셜지표 | 불필요 | 30/분 | [링크](https://www.coingecko.com/en/api/documentation) |
| **Alternative.me** | Fear & Greed Index | 불필요 | 60/분 | [링크](https://alternative.me/crypto/api/) |
| **CryptoPanic** | 뉴스 센티멘트 | 필요 (무료) | 5/분 | [링크](https://cryptopanic.com/developers/api/) |
| **CoinMarketCap** | 시세, 글로벌 | 필요 | 30/분 | [링크](https://coinmarketcap.com/api/) |

---

## 개발

### 테스트

```bash
uv run pytest
uv run pytest -v  # 상세
```

### 린트 & 포맷

```bash
uv run ruff check .   # 린트
uv run ruff format .  # 포맷
uv run mypy src       # 타입 체크
```

### TensorBoard

```bash
uv run tensorboard --logdir runs
```

---

## 관련 문서

- [ROADMAP.md](ROADMAP.md) - 개발 로드맵 및 진행 상황
- [docs/MODELS.md](docs/MODELS.md) - AI 모델 아키텍처 상세 설명
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - 시스템 아키텍처 및 플로우 차트

---

## 라이선스

MIT
