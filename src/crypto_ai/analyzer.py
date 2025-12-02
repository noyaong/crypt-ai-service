"""PyTorch 기반 차트 분석 모델"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from crypto_ai.models import TrendDirection


def get_device() -> torch.device:
    """
    최적의 연산 디바이스 자동 선택

    우선순위:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)
    """
    # NVIDIA GPU
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon GPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    # CPU fallback
    return torch.device("cpu")


DEVICE = get_device()


class ChartAnalyzer(nn.Module):
    """LSTM 기반 가격 예측 및 패턴 분석"""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,  # [price, volume, rsi, macd, bb_position]
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3),  # [상승, 하락, 횡보] 확률
        )

        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # 마지막 타임스텝
        logits = self.fc(last_hidden)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI (Relative Strength Index) 계산"""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = pd.Series(gains).rolling(window=period, min_periods=1).mean().to_numpy()
        avg_loss = pd.Series(losses).rolling(window=period, min_periods=1).mean().to_numpy()

        rs = np.divide(
            avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_gain)
        )
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def compute_macd(
        prices: np.ndarray, fast: int = 12, slow: int = 26
    ) -> np.ndarray:
        """MACD (Moving Average Convergence Divergence) 계산"""
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().to_numpy()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().to_numpy()
        return ema_fast - ema_slow

    @staticmethod
    def compute_bollinger_position(
        prices: np.ndarray, period: int = 20
    ) -> np.ndarray:
        """볼린저 밴드 내 위치 (0~1)"""
        rolling = pd.Series(prices).rolling(window=period, min_periods=1)
        middle = rolling.mean().to_numpy()
        std = rolling.std().to_numpy()
        std = np.nan_to_num(std, nan=0.0)

        upper = middle + 2 * std
        lower = middle - 2 * std

        bandwidth = upper - lower
        position = np.divide(
            prices - lower,
            bandwidth,
            where=bandwidth != 0,
            out=np.full_like(prices, 0.5),
        )
        return np.clip(position, 0, 1)

    def prepare_features(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> torch.Tensor:
        """특성 벡터 준비"""
        # 정규화
        price_std = prices.std()
        vol_std = volumes.std()

        price_norm = (prices - prices.mean()) / (price_std + 1e-8)
        vol_norm = (volumes - volumes.mean()) / (vol_std + 1e-8)

        rsi = self.compute_rsi(prices) / 100.0  # 0~1로 정규화
        macd = self.compute_macd(prices)
        macd_std = macd.std()
        macd_norm = (macd - macd.mean()) / (macd_std + 1e-8)
        bb_pos = self.compute_bollinger_position(prices)

        features = np.stack([price_norm, vol_norm, rsi, macd_norm, bb_pos], axis=-1)
        return torch.tensor(features, dtype=torch.float32, device=DEVICE)

    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> dict:
        """차트 분석 수행"""
        self.eval()

        features = self.prepare_features(prices, volumes)
        features = features.unsqueeze(0)  # 배치 차원 추가

        with torch.no_grad():
            probs = self(features)

        probs_np = probs.cpu().numpy()[0]
        direction_idx = int(np.argmax(probs_np))
        directions = [TrendDirection.BULLISH, TrendDirection.BEARISH, TrendDirection.NEUTRAL]

        return {
            "trend": directions[direction_idx],
            "confidence": float(probs_np[direction_idx]),
            "probabilities": {
                "상승": float(probs_np[0]),
                "하락": float(probs_np[1]),
                "횡보": float(probs_np[2]),
            },
            "indicators": {
                "rsi": float(self.compute_rsi(prices)[-1]),
                "macd": float(self.compute_macd(prices)[-1]),
                "bollinger_position": float(self.compute_bollinger_position(prices)[-1]),
            },
        }
