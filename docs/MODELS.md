# AI 모델 아키텍처 가이드

이 문서는 Crypto AI Service에서 사용하는 두 가지 딥러닝 모델의 원리, 장단점, 사용법을 설명합니다.

## 목차

- [모델 비교 요약](#모델-비교-요약)
- [LSTM 모델](#lstm-모델)
- [Transformer 모델](#transformer-모델)
- [멀티태스크 학습](#멀티태스크-학습)
- [모델 선택 가이드](#모델-선택-가이드)

---

## 모델 비교 요약

| 항목 | LSTM | Transformer |
|------|------|-------------|
| **파일** | `analyzer.py` | `transformer.py` |
| **파라미터** | ~50K | ~157K |
| **처리 방식** | 순차적 (Sequential) | 병렬 (Parallel) |
| **장거리 의존성** | 약함 | 강함 (Attention) |
| **학습 속도** | 느림 | 빠름 (MPS 최적화) |
| **멀티태스크** | ❌ | ✅ |
| **해석 가능성** | 낮음 | 높음 (Attention 시각화) |
| **메모리 사용** | 낮음 | 높음 |

---

## LSTM 모델

### 원리

LSTM(Long Short-Term Memory)은 순환 신경망(RNN)의 한 종류로, 시계열 데이터를 **순차적으로** 처리합니다.

```
t=1 → t=2 → t=3 → ... → t=60 → 예측
 ↓      ↓      ↓           ↓
[h1] → [h2] → [h3] → ... → [h60] → FC → 출력
```

#### 핵심 메커니즘

1. **Forget Gate**: 이전 정보 중 버릴 것 결정
2. **Input Gate**: 새로운 정보 중 저장할 것 결정
3. **Output Gate**: 현재 출력할 정보 결정

```python
# 간략화된 LSTM 수식
forget = sigmoid(W_f · [h_prev, x])      # 잊을 비율
input = sigmoid(W_i · [h_prev, x])       # 저장할 비율
candidate = tanh(W_c · [h_prev, x])      # 후보 정보
cell = forget * cell_prev + input * candidate  # 셀 상태 업데이트
output = sigmoid(W_o · [h_prev, x]) * tanh(cell)  # 출력
```

### 아키텍처 (analyzer.py)

```
입력 (60, 9)
    ↓
LSTM Layer 1 (hidden=64)
    ↓ dropout=0.2
LSTM Layer 2 (hidden=64)
    ↓
마지막 hidden state (64,)
    ↓
FC: 64 → 32 → 3
    ↓
Softmax
    ↓
출력: [하락, 횡보, 상승] 확률
```

### 장점

- **단순함**: 구현과 이해가 쉬움
- **적은 파라미터**: 메모리 효율적
- **안정적 학습**: 오래된 기술로 검증됨

### 단점

- **장거리 의존성 문제**: 60시간 전 정보가 희석됨
- **순차 처리**: 병렬화 어려움, 느린 학습
- **기울기 소실**: 깊은 시퀀스에서 학습 어려움

### 사용법

```bash
# 학습
uv run python scripts/train.py --symbol BTC --days 180 --epochs 50

# 예측
uv run crypto-ai predict BTC --model lstm
```

---

## Transformer 모델

### 원리

Transformer는 **Self-Attention** 메커니즘을 사용하여 시퀀스의 모든 위치를 동시에 참조합니다.

```
t=1   t=2   t=3   ...  t=60
 ↓     ↓     ↓          ↓
[  Attention: 모든 시점 간 관계 계산  ]
 ↓     ↓     ↓          ↓
[      Global Pooling              ]
              ↓
           예측
```

#### Self-Attention 메커니즘

"60시간 데이터 중 어느 시점이 현재 예측에 중요한가?"를 학습합니다.

```python
# Attention 수식
Q = x @ W_q  # Query: "무엇을 찾고 있나?"
K = x @ W_k  # Key: "내가 가진 정보는?"
V = x @ W_v  # Value: "실제 정보"

# Attention Score: Q와 K의 유사도
scores = (Q @ K.T) / sqrt(d_k)
attention = softmax(scores)

# 가중 합
output = attention @ V
```

#### Multi-Head Attention

여러 개의 Attention을 병렬로 수행하여 다양한 관점에서 패턴을 포착합니다.

```
Head 1: 단기 변동 패턴 학습
Head 2: 거래량 급증 패턴 학습
Head 3: 추세 전환 패턴 학습
Head 4: 지지/저항 패턴 학습
         ↓
      Concat → Linear → 출력
```

### 아키텍처 (transformer.py)

```
입력 (60, 9)
    ↓
Linear Projection: 9 → 64
    ↓
Positional Encoding (시간 순서 정보)
    ↓
┌─────────────────────────────────┐
│  Transformer Encoder Layer × 3  │
│  ├─ Multi-Head Attention (4 heads)
│  ├─ Layer Norm + Residual
│  ├─ Feed Forward (64 → 256 → 64)
│  └─ Layer Norm + Residual
└─────────────────────────────────┘
    ↓
Global Average Pooling
    ↓
┌─────────────────────────────────┐
│  Multi-Task Heads               │
│  ├─ Direction: 64 → 32 → 3     │
│  ├─ Volatility: 64 → 32 → 1    │
│  └─ Volume: 64 → 32 → 1        │
└─────────────────────────────────┘
    ↓
출력: 방향 확률 + 변동성 + 거래량
```

### 장점

- **장거리 의존성**: 60시간 전 정보도 직접 참조 가능
- **병렬 처리**: MPS/GPU에서 빠른 학습
- **해석 가능**: Attention 가중치로 중요 시점 확인
- **멀티태스크**: 여러 예측을 동시에 학습

### 단점

- **많은 파라미터**: 더 많은 메모리 필요
- **데이터 요구량**: 충분한 데이터 필요
- **과적합 위험**: 작은 데이터셋에서 주의 필요

### 사용법

```bash
# 학습 (멀티태스크)
uv run python scripts/train_transformer.py \
    --symbol BTC \
    --days 365 \
    --epochs 100 \
    --multi-task

# 예측
uv run crypto-ai predict BTC --model transformer
```

### Attention 시각화

학습 후 `checkpoints/transformer/attention.png`에서 Attention 패턴을 확인할 수 있습니다.

```python
# 코드로 시각화
from crypto_ai.transformer import visualize_attention

outputs = model(x, return_attention=True)
visualize_attention(outputs["attention"], save_path="attention.png")
```

---

## 멀티태스크 학습

Transformer 모델은 세 가지 태스크를 동시에 학습합니다:

### 1. 가격 방향 예측 (분류)

- **출력**: 상승 / 하락 / 횡보 확률
- **손실함수**: CrossEntropyLoss (클래스 가중치 적용)
- **임계값**: ±0.5% 변동을 기준으로 분류

### 2. 변동성 예측 (회귀)

- **출력**: 다음 봉의 (고가-저가)/종가 비율
- **손실함수**: MSELoss
- **해석**: 양수=높은 변동성, 음수=낮은 변동성

### 3. 거래량 변화 예측 (회귀)

- **출력**: 다음 봉의 거래량 변화율
- **손실함수**: MSELoss
- **해석**: 양수=거래량 증가, 음수=거래량 감소

### 손실 함수 조합

```python
total_loss = 1.0 * direction_loss    # 가격 방향 (주요)
           + 0.3 * volatility_loss   # 변동성 (보조)
           + 0.2 * volume_loss       # 거래량 (보조)
```

### 장점

- **상호 보완**: 거래량/변동성 정보가 방향 예측에 도움
- **정규화 효과**: 과적합 방지
- **추가 정보**: 단순 방향 외에 시장 상황 파악

---

## 모델 선택 가이드

### 언제 LSTM?

- ✅ 빠른 프로토타이핑
- ✅ 제한된 컴퓨팅 자원
- ✅ 작은 데이터셋 (< 1000개)
- ✅ 단순한 방향 예측만 필요

### 언제 Transformer?

- ✅ 정확도가 중요할 때
- ✅ 충분한 데이터 (> 5000개)
- ✅ 변동성/거래량 예측도 필요
- ✅ 예측 근거를 분석하고 싶을 때
- ✅ GPU 가속 (CUDA/MPS) 또는 CPU

### 권장 설정

**테스트/개발:**
```bash
# LSTM, 빠른 확인
uv run python scripts/train.py --days 60 --epochs 10
```

**프로덕션:**
```bash
# Transformer, 멀티태스크
uv run python scripts/train_transformer.py \
    --days 365 \
    --epochs 100 \
    --multi-task \
    --d-model 128 \
    --num-heads 8
```

---

## 기술적 지표 입력

두 모델 모두 동일한 9개 특성을 입력으로 사용합니다:

| 특성 | 설명 | 범위 |
|------|------|------|
| `returns` | 가격 변화율 | 정규화 |
| `log_returns` | 로그 수익률 | 정규화 |
| `rsi` | RSI (14) | 0~1 |
| `macd` | MACD | 정규화 |
| `macd_signal` | MACD 시그널 | 정규화 |
| `macd_hist` | MACD 히스토그램 | 정규화 |
| `bb_position` | 볼린저밴드 위치 | 0~1 |
| `volume_ratio` | 거래량/이동평균 | 정규화 |
| `atr` | ATR (14) | 정규화 |

---

## 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 원 논문
- [LSTM 이해하기](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch CUDA](https://pytorch.org/docs/stable/cuda.html)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
