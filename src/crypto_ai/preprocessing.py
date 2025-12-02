"""데이터 전처리 파이프라인"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from crypto_ai.data_sources import BinanceClient, UnifiedDataCollector, AlternativeMeClient, CoinGeckoClient


# 모델 입력 특성 정의 (13개)
FEATURE_COLUMNS = [
    # 기술적 지표 (9개)
    "returns", "log_returns",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_position",
    "volume_ratio", "atr",
    # 시장 데이터 (4개)
    "fear_greed", "fear_greed_ma7", "fear_greed_change",
    "btc_dominance",
]

INPUT_SIZE = len(FEATURE_COLUMNS)  # 13


@dataclass
class DataConfig:
    """데이터 설정"""
    symbol: str = "BTC"
    interval: str = "1h"
    days: int = 365
    sequence_length: int = 60  # 60시간 = 2.5일
    prediction_horizon: int = 1  # 다음 1봉 예측
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    multi_task: bool = False  # 멀티태스크 학습 활성화


class CryptoDataset(Dataset):
    """PyTorch Dataset for crypto OHLCV data"""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 60,
        volatility_targets: np.ndarray | None = None,
        volume_targets: np.ndarray | None = None,
    ):
        """
        Args:
            features: (N, num_features) 형태의 특성 배열
            labels: (N,) 형태의 레이블 배열 (0=하락, 1=횡보, 2=상승)
            sequence_length: 시퀀스 길이
            volatility_targets: (N,) 변동성 타겟 (멀티태스크용)
            volume_targets: (N,) 거래량 변화 타겟 (멀티태스크용)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sequence_length = sequence_length

        self.multi_task = volatility_targets is not None
        if self.multi_task:
            self.volatility_targets = torch.tensor(volatility_targets, dtype=torch.float32)
            self.volume_targets = torch.tensor(volume_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.features[idx:idx + self.sequence_length]
        target_idx = idx + self.sequence_length

        targets = {"direction": self.labels[target_idx]}

        if self.multi_task:
            targets["volatility"] = self.volatility_targets[target_idx].unsqueeze(0)
            targets["volume"] = self.volume_targets[target_idx].unsqueeze(0)

        return x, targets


class DataPipeline:
    """데이터 수집, 전처리, 분할 파이프라인"""

    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()
        self.collector = UnifiedDataCollector()
        self.fng_client = AlternativeMeClient()
        self.coingecko_client = CoinGeckoClient()
        self.scaler_params: dict = {}
        self._market_data_cache: dict = {}  # 시장 데이터 캐시

    def fetch_data(
        self,
        symbol: str | None = None,
        interval: str | None = None,
        days: int | None = None,
    ) -> pd.DataFrame:
        """Binance에서 OHLCV 데이터 수집"""
        symbol = symbol or self.config.symbol
        interval = interval or self.config.interval
        days = days or self.config.days

        df = self.collector.get_ohlcv(
            symbol=symbol,
            source="binance",
            interval=interval,
            days=days,
        )

        if df.empty:
            raise ValueError(f"No data fetched for {symbol}")

        print(f"Fetched {len(df)} rows for {symbol} ({interval}, {days} days)")
        return df

    def fetch_market_data(self, days: int | None = None) -> dict:
        """
        시장 전반 데이터 수집 (Fear & Greed, BTC 도미넌스 등)

        Returns:
            {
                'fear_greed': DataFrame with date index and value column,
                'btc_dominance': float,
                'total_market_cap': float,
            }
        """
        days = days or self.config.days

        # Fear & Greed Index (히스토리)
        try:
            fng_df = self.fng_client.get_fear_greed_history(days=min(days, 365))
            if not fng_df.empty and 'timestamp' in fng_df.columns:
                # timestamp가 이미 숫자인 경우와 문자열인 경우 처리
                fng_df['timestamp'] = pd.to_numeric(fng_df['timestamp'], errors='coerce')
                fng_df = fng_df.dropna(subset=['timestamp'])
                fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'], unit='s', errors='coerce')
                fng_df = fng_df.dropna(subset=['timestamp'])
                fng_df['value'] = pd.to_numeric(fng_df['value'], errors='coerce').fillna(50)
                fng_df = fng_df.set_index('timestamp').sort_index()
        except Exception as e:
            print(f"Warning: Failed to fetch Fear & Greed history: {e}")
            fng_df = pd.DataFrame()

        # 글로벌 마켓 데이터 (BTC 도미넌스 등)
        try:
            global_data = self.coingecko_client.get_global_data()
            btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 50.0)
            total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
        except Exception as e:
            print(f"Warning: Failed to fetch global market data: {e}")
            btc_dominance = 50.0
            total_market_cap = 0

        self._market_data_cache = {
            'fear_greed_df': fng_df,
            'btc_dominance': btc_dominance,
            'total_market_cap': total_market_cap,
        }

        return self._market_data_cache

    def _merge_fear_greed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLCV 데이터프레임에 Fear & Greed 값 병합
        (일봉 데이터를 시간봉에 forward fill)
        """
        if 'fear_greed_df' not in self._market_data_cache:
            self.fetch_market_data()

        fng_df = self._market_data_cache.get('fear_greed_df', pd.DataFrame())

        if fng_df.empty:
            # 기본값 50 (중립)
            df['fear_greed'] = 50.0
            return df

        # OHLCV의 timestamp를 날짜로 변환하여 매칭
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.normalize()

        # Fear & Greed도 날짜로 정규화
        fng_df = fng_df.copy()
        fng_df['date'] = fng_df.index.normalize()
        fng_daily = fng_df.groupby('date')['value'].first()

        # 날짜 기준 병합
        df['fear_greed'] = df['date'].map(fng_daily)

        # Forward fill (주말 등 데이터 없는 날)
        df['fear_greed'] = df['fear_greed'].ffill().bfill().fillna(50.0)

        df = df.drop(columns=['date'])
        return df

    def compute_features(self, df: pd.DataFrame, include_market_data: bool = True) -> pd.DataFrame:
        """기술적 지표 계산"""
        result = df.copy()

        # 가격 변화율
        result["returns"] = result["close"].pct_change()
        result["log_returns"] = np.log(result["close"] / result["close"].shift(1))

        # RSI
        result["rsi"] = self._compute_rsi(result["close"].values)

        # MACD
        result["macd"], result["macd_signal"] = self._compute_macd(result["close"].values)
        result["macd_hist"] = result["macd"] - result["macd_signal"]

        # 볼린저 밴드
        result["bb_upper"], result["bb_middle"], result["bb_lower"] = self._compute_bollinger(
            result["close"].values
        )
        result["bb_position"] = (result["close"] - result["bb_lower"]) / (
            result["bb_upper"] - result["bb_lower"] + 1e-8
        )

        # 이동평균
        result["sma_20"] = result["close"].rolling(20).mean()
        result["sma_50"] = result["close"].rolling(50).mean()
        result["ema_12"] = result["close"].ewm(span=12, adjust=False).mean()
        result["ema_26"] = result["close"].ewm(span=26, adjust=False).mean()

        # 거래량 지표
        result["volume_sma"] = result["volume"].rolling(20).mean()
        result["volume_ratio"] = result["volume"] / (result["volume_sma"] + 1e-8)

        # ATR (Average True Range)
        result["atr"] = self._compute_atr(
            result["high"].values,
            result["low"].values,
            result["close"].values,
        )

        # 시장 데이터 추가 (Fear & Greed, BTC 도미넌스)
        if include_market_data:
            result = self._add_market_features(result)

        # NaN 제거
        result = result.dropna().reset_index(drop=True)

        return result

    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시장 전반 특성 추가 (Fear & Greed, BTC 도미넌스 등)"""
        result = df.copy()

        # Fear & Greed Index 병합
        result = self._merge_fear_greed(result)

        # Fear & Greed 이동평균 (7일)
        result["fear_greed_ma7"] = result["fear_greed"].rolling(window=24*7, min_periods=1).mean()

        # Fear & Greed 변화율
        result["fear_greed_change"] = result["fear_greed"].pct_change(periods=24).fillna(0)

        # BTC 도미넌스 (현재 값 - 히스토리 없으므로 상수)
        btc_dom = self._market_data_cache.get('btc_dominance', 50.0)
        result["btc_dominance"] = btc_dom

        print(f"Market features added: Fear & Greed, BTC Dominance ({btc_dom:.1f}%)")

        return result

    def create_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.005,
    ) -> np.ndarray:
        """
        레이블 생성 (다음 N봉 후 가격 방향)

        Args:
            df: OHLCV + features DataFrame
            horizon: 예측 기간 (봉 개수)
            threshold: 횡보 판단 임계값 (0.5% = 0.005)

        Returns:
            labels: 0=하락, 1=횡보, 2=상승
        """
        future_returns = df["close"].shift(-horizon) / df["close"] - 1

        labels = np.where(
            future_returns > threshold, 2,  # 상승
            np.where(future_returns < -threshold, 0, 1)  # 하락 or 횡보
        )

        return labels

    def create_multi_task_targets(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        멀티태스크 타겟 생성 (변동성 + 거래량 변화)

        Args:
            df: OHLCV + features DataFrame
            horizon: 예측 기간 (봉 개수)

        Returns:
            (volatility_targets, volume_targets)
        """
        # 변동성: 다음 N봉의 고가-저가 범위 / 종가
        future_high = df["high"].shift(-horizon)
        future_low = df["low"].shift(-horizon)
        future_close = df["close"].shift(-horizon)
        volatility = (future_high - future_low) / (future_close + 1e-8)

        # 거래량 변화율: 다음 N봉의 거래량 변화
        future_volume = df["volume"].shift(-horizon)
        current_volume = df["volume"]
        volume_change = (future_volume - current_volume) / (current_volume + 1e-8)

        # 정규화 (Z-score)
        volatility_norm = (volatility - volatility.mean()) / (volatility.std() + 1e-8)
        volume_change_norm = (volume_change - volume_change.mean()) / (volume_change.std() + 1e-8)

        return volatility_norm.values, volume_change_norm.values

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        fit: bool = True,
    ) -> np.ndarray:
        """
        특성 정규화 (Z-score)

        Args:
            df: DataFrame
            feature_columns: 정규화할 컬럼 목록
            fit: True면 파라미터 저장, False면 저장된 파라미터 사용
        """
        features = df[feature_columns].values

        if fit:
            self.scaler_params = {
                "mean": features.mean(axis=0),
                "std": features.std(axis=0) + 1e-8,
            }

        normalized = (features - self.scaler_params["mean"]) / self.scaler_params["std"]
        return normalized

    def prepare_dataset(
        self,
        symbol: str | None = None,
        days: int | None = None,
        multi_task: bool | None = None,
    ) -> tuple[CryptoDataset, CryptoDataset, CryptoDataset]:
        """
        전체 파이프라인 실행: 데이터 수집 -> 전처리 -> 분할

        Args:
            symbol: 심볼 (기본값: config.symbol)
            days: 기간 (기본값: config.days)
            multi_task: 멀티태스크 학습 활성화 (기본값: config.multi_task)

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        multi_task = multi_task if multi_task is not None else self.config.multi_task

        # 1. 데이터 수집
        df = self.fetch_data(symbol=symbol, days=days)

        # 2. 특성 계산
        df = self.compute_features(df)
        print(f"Computed features: {len(df)} rows")

        # 3. 레이블 생성
        labels = self.create_labels(df, horizon=self.config.prediction_horizon)

        # 4. 멀티태스크 타겟 생성
        volatility_targets = None
        volume_targets = None
        if multi_task:
            volatility_targets, volume_targets = self.create_multi_task_targets(
                df, horizon=self.config.prediction_horizon
            )
            print("Multi-task targets created (volatility + volume)")

        # 5. 특성 선택 및 정규화 (13개 특성)
        feature_columns = FEATURE_COLUMNS

        # Train/Val/Test 분할 인덱스
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        # Train 데이터로 정규화 파라미터 학습
        train_df = df.iloc[:train_end]
        features_train = self.normalize_features(train_df, feature_columns, fit=True)
        labels_train = labels[:train_end]

        # Val/Test는 Train 파라미터로 정규화
        val_df = df.iloc[train_end:val_end]
        features_val = self.normalize_features(val_df, feature_columns, fit=False)
        labels_val = labels[train_end:val_end]

        test_df = df.iloc[val_end:]
        features_test = self.normalize_features(test_df, feature_columns, fit=False)
        labels_test = labels[val_end:]

        # Dataset 생성
        seq_len = self.config.sequence_length

        if multi_task:
            train_dataset = CryptoDataset(
                features_train, labels_train, seq_len,
                volatility_targets[:train_end], volume_targets[:train_end]
            )
            val_dataset = CryptoDataset(
                features_val, labels_val, seq_len,
                volatility_targets[train_end:val_end], volume_targets[train_end:val_end]
            )
            test_dataset = CryptoDataset(
                features_test, labels_test, seq_len,
                volatility_targets[val_end:], volume_targets[val_end:]
            )
        else:
            train_dataset = CryptoDataset(features_train, labels_train, seq_len)
            val_dataset = CryptoDataset(features_val, labels_val, seq_len)
            test_dataset = CryptoDataset(features_test, labels_test, seq_len)

        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # 레이블 분포 출력
        self._print_label_distribution(labels_train, "Train")
        self._print_label_distribution(labels_val, "Val")
        self._print_label_distribution(labels_test, "Test")

        return train_dataset, val_dataset, test_dataset

    def get_dataloaders(
        self,
        train_dataset: CryptoDataset,
        val_dataset: CryptoDataset,
        test_dataset: CryptoDataset,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """DataLoader 생성"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader, test_loader

    # ========== Helper Methods ==========

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI 계산"""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = pd.Series(gains).rolling(window=period, min_periods=1).mean().values
        avg_loss = pd.Series(losses).rolling(window=period, min_periods=1).mean().values

        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_gain))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """MACD 계산"""
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().values
        macd = ema_fast - ema_slow
        macd_signal = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
        return macd, macd_signal

    @staticmethod
    def _compute_bollinger(
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """볼린저 밴드 계산"""
        rolling = pd.Series(prices).rolling(window=period, min_periods=1)
        middle = rolling.mean().values
        std = rolling.std().values
        std = np.nan_to_num(std, nan=0.0)

        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    @staticmethod
    def _compute_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """ATR (Average True Range) 계산"""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            )
        )
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
        return atr

    @staticmethod
    def _print_label_distribution(labels: np.ndarray, name: str) -> None:
        """레이블 분포 출력"""
        unique, counts = np.unique(labels[~np.isnan(labels)].astype(int), return_counts=True)
        total = counts.sum()
        label_names = {0: "하락", 1: "횡보", 2: "상승"}

        dist = ", ".join([f"{label_names.get(u, u)}: {c} ({c/total:.1%})" for u, c in zip(unique, counts)])
        print(f"  {name} labels: {dist}")


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    # 기본 설정으로 파이프라인 실행
    config = DataConfig(
        symbol="BTC",
        interval="1h",
        days=180,  # 6개월
        sequence_length=60,
    )

    pipeline = DataPipeline(config)
    train_ds, val_ds, test_ds = pipeline.prepare_dataset()

    # DataLoader 생성
    train_loader, val_loader, test_loader = pipeline.get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=32
    )

    # 샘플 확인
    x, y = next(iter(train_loader))
    print(f"\nBatch shape: x={x.shape}, y={y.shape}")
    print(f"Device: x on {x.device}")
