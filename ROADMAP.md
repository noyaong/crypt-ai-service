# Crypto AI Service - Claude Code 작업 계획

## 프로젝트 개요

MacBook MPS(Metal) + PyTorch 기반 암호화폐 AI 분석 서비스

### 현재 완성된 기능

**데이터 수집**
- [x] CoinMarketCap API 클라이언트
- [x] Binance API 클라이언트 (무료 OHLCV, 무제한)
- [x] CoinGecko API 클라이언트 (무료)
- [x] Alternative.me API 클라이언트 (Fear & Greed Index)
- [x] 통합 데이터 수집기 (UnifiedDataCollector)

**AI 모델**
- [x] PyTorch LSTM 차트 분석 모델
- [x] **Transformer 예측 모델 (Multi-head Attention)**
- [x] **멀티태스크 학습 (가격 방향 + 변동성 + 거래량)**
- [x] 기술적 지표 (RSI, MACD, 볼린저밴드, ATR)

**학습 파이프라인**
- [x] 데이터 전처리 파이프라인 (preprocessing.py)
- [x] LSTM 학습 스크립트 (scripts/train.py)
- [x] Transformer 학습 스크립트 (scripts/train_transformer.py)
- [x] 체크포인트 저장/로드
- [x] TensorBoard 학습 로그
- [x] Attention 시각화

**인터페이스**
- [x] CLI 명령어 (crypto-ai)
- [x] FastAPI REST API
- [x] 온체인 인사이트 분석기

---

## 📊 데이터 소스 비교

| 기능 | CoinMarketCap | Binance | CoinGecko |
|------|---------------|---------|-----------|
| **가격** | ✅ 무료 | ✅ 무료 | ✅ 무료 |
| **OHLCV 히스토리** | ❌ $79/월 | ✅ **무료** | ✅ 무료 (제한) |
| **인터벌** | 일봉만 | 1분~1개월 | 4시간~4일 |
| **히스토리 기간** | 2013~ | 2017~ | 2013~ |
| **Rate Limit** | 30/분 | 1200/분 | 30/분 |
| **API 키** | 필요 | 불필요 | 불필요 |
| **시가총액** | ✅ | ❌ | ✅ |
| **글로벌 메트릭** | ✅ | ❌ | ✅ |

### 💡 권장 조합

```
시세/메타데이터: CoinMarketCap (무료) 또는 CoinGecko
OHLCV 학습 데이터: Binance API (무료, 무제한)
실시간 스트리밍: Binance WebSocket
```

---

## Phase 1: 기반 완성 ✅

### 1.1 환경 설정

```bash
cd /Users/jsnoh/workspace/crypto-ai-service
uv sync --all-extras
cp .env.example .env
# .env에 CMC_API_KEY 입력 (선택사항)
```

### 1.2 동작 확인

```bash
# MPS 확인
uv run crypto-ai check

# 테스트
uv run pytest

# 시세 조회
uv run crypto-ai price BTC ETH AVAX

# API 서버
uv run uvicorn crypto_ai.api:app --reload
```

---

## Phase 2: 모델 학습 파이프라인 ✅ 완료

### 2.1 데이터 수집 (Binance 무료 API 활용)

```python
from crypto_ai import UnifiedDataCollector

collector = UnifiedDataCollector()

# 1년치 1시간봉 데이터 (무료!)
df = collector.get_ohlcv(
    symbol="BTC",
    source="binance",
    interval="1h",
    days=365
)
print(f"수집된 데이터: {len(df)} rows")
```

### 2.2 학습 파이프라인 ✅

```bash
# LSTM 모델 학습
uv run python scripts/train.py --symbol BTC --days 365 --epochs 100

# Transformer 모델 학습 (권장)
uv run python scripts/train_transformer.py \
    --symbol BTC \
    --days 365 \
    --epochs 100 \
    --multi-task \
    --d-model 64 \
    --num-heads 4 \
    --num-layers 3

# TensorBoard 모니터링
uv run tensorboard --logdir runs
```

**구현 완료:**
- [x] 데이터 전처리 파이프라인 (`preprocessing.py`)
- [x] Train/Val/Test 분할 (70/15/15)
- [x] LSTM 학습 스크립트 (`scripts/train.py`)
- [x] 체크포인트 저장/로드 (`checkpoints/`)
- [x] TensorBoard 학습 로그 (`runs/`)

### 2.3 Transformer 모델 ✅

**아키텍처:**
- Multi-head Self-Attention (4 heads, 3 layers)
- Positional Encoding
- 멀티태스크 학습 (가격 방향 + 변동성 + 거래량)
- ~157K 파라미터

**구현 완료:**
- [x] Transformer 모델 (`transformer.py`)
- [x] Attention 시각화 (`checkpoints/transformer/attention.png`)
- [x] 학습 스크립트 (`scripts/train_transformer.py`)

---

## Phase 3: 데이터 소스 확장 🎯 현재 단계

### 3.1 Arkham Intelligence 연동

- [ ] API 클라이언트 구현
- [ ] 지갑 추적 기능
- [ ] 기관 매집/매도 감지
- [ ] 대형 거래 알림

### 3.2 추가 데이터 소스

- [ ] Binance 실시간 WebSocket
- [ ] Glassnode 온체인 메트릭
- [x] **Fear & Greed Index API (Alternative.me)** ✅

---

## Phase 4: 인터페이스

### 4.1 Generative UI (Next.js)

- [ ] AI 대화형 분석 인터페이스
- [ ] Vercel AI SDK 연동
- [ ] 실시간 차트 시각화
- [ ] 포트폴리오 추적

### 4.2 알림 시스템

- [ ] Slack/Discord 웹훅
- [ ] 가격 알림
- [ ] 이상 거래 감지

---

## Phase 5: 배포 & 운영

### 5.1 컨테이너화

- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] GitHub Actions CI/CD

### 5.2 모니터링

- [ ] Prometheus 메트릭
- [ ] Grafana 대시보드
- [ ] 에러 추적 (Sentry)

---

## 주요 파일 구조

```
/Users/jsnoh/workspace/crypto-ai-service/
├── pyproject.toml          # uv 프로젝트 설정
├── README.md               # 사용 가이드
├── ROADMAP.md              # 이 파일
├── .env.example            # 환경변수 템플릿
├── .gitignore
├── src/crypto_ai/
│   ├── __init__.py         # 메인 서비스 클래스
│   ├── cli.py              # CLI (crypto-ai 명령어)
│   ├── api.py              # FastAPI 서버
│   ├── client.py           # CoinMarketCap 클라이언트
│   ├── data_sources.py     # ✨ Binance/CoinGecko/Alternative.me 클라이언트
│   ├── analyzer.py         # PyTorch LSTM 차트 분석 모델
│   ├── transformer.py      # ✨ Transformer 예측 모델
│   ├── preprocessing.py    # ✨ 데이터 전처리 파이프라인
│   ├── insight.py          # 인사이트 생성기
│   └── models.py           # 데이터 모델
├── scripts/
│   ├── train.py            # ✨ LSTM 모델 학습 스크립트
│   └── train_transformer.py # ✨ Transformer 모델 학습 스크립트
├── checkpoints/            # 학습된 모델 체크포인트
├── runs/                   # TensorBoard 로그
└── tests/
    ├── test_analyzer.py
    └── test_client.py
```

---

## 🚀 Quick Start

### 1. 환경 설정

```bash
cd /Users/jsnoh/workspace/crypto-ai-service
uv sync --all-extras
cp .env.example .env  # CMC_API_KEY 설정 (선택)
```

### 2. 데이터 수집 테스트 (API 키 불필요)

```bash
# Binance OHLCV
uv run python -c "
from crypto_ai import BinanceClient
client = BinanceClient()
df = client.get_klines('BTCUSDT', '1h', limit=100)
print(df.tail())
"

# Fear & Greed Index
uv run python -c "
from crypto_ai import AlternativeMeClient
client = AlternativeMeClient()
result = client.get_fear_greed_index()
print(f\"Fear & Greed: {result['value']} ({result['value_classification']})\")"
```

### 3. 모델 학습

```bash
# LSTM 모델 (빠른 테스트)
uv run python scripts/train.py --symbol BTC --days 60 --epochs 10

# Transformer 모델 (권장, 멀티태스크)
uv run python scripts/train_transformer.py --symbol BTC --days 90 --epochs 20 --multi-task

# TensorBoard로 학습 모니터링
uv run tensorboard --logdir runs
```

### 4. CLI 사용

```bash
uv run crypto-ai check          # MPS 상태 확인
uv run crypto-ai price BTC ETH  # 시세 조회
uv run crypto-ai market         # 시장 인사이트
```

### 5. API 서버

```bash
uv run uvicorn crypto_ai.api:app --reload
# http://localhost:8000/docs
```

---

## 참고 자료

- [Binance API Docs](https://binance-docs.github.io/apidocs/spot/en/)
- [CoinGecko API Docs](https://www.coingecko.com/en/api/documentation)
- [CoinMarketCap API Docs](https://coinmarketcap.com/api/documentation/v1/)
- [Alternative.me Crypto API](https://alternative.me/crypto/api/) - Fear & Greed Index
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [uv Documentation](https://docs.astral.sh/uv/)
