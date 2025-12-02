"""ChartAnalyzer 테스트"""

import numpy as np
import pytest
import torch

from crypto_ai.analyzer import ChartAnalyzer, get_device
from crypto_ai.models import TrendDirection


class TestGetDevice:
    """get_device 테스트"""

    def test_returns_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("mps", "cpu")


class TestChartAnalyzer:
    """ChartAnalyzer 테스트"""

    @pytest.fixture
    def analyzer(self):
        return ChartAnalyzer()

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        prices = np.random.randn(100).cumsum() + 100
        volumes = np.abs(np.random.randn(100)) * 1000000
        return prices, volumes

    def test_initialization(self, analyzer):
        assert isinstance(analyzer, torch.nn.Module)
        assert analyzer.hidden_size == 64
        assert analyzer.num_layers == 2

    def test_compute_rsi(self, sample_data):
        prices, _ = sample_data
        rsi = ChartAnalyzer.compute_rsi(prices)

        assert len(rsi) == len(prices)
        assert all(0 <= v <= 100 for v in rsi)

    def test_compute_macd(self, sample_data):
        prices, _ = sample_data
        macd = ChartAnalyzer.compute_macd(prices)

        assert len(macd) == len(prices)
        assert not np.any(np.isnan(macd))

    def test_compute_bollinger_position(self, sample_data):
        prices, _ = sample_data
        bb_pos = ChartAnalyzer.compute_bollinger_position(prices)

        assert len(bb_pos) == len(prices)
        assert all(0 <= v <= 1 for v in bb_pos)

    def test_prepare_features(self, analyzer, sample_data):
        prices, volumes = sample_data
        features = analyzer.prepare_features(prices, volumes)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (100, 5)
        assert features.device.type == get_device().type

    def test_forward(self, analyzer, sample_data):
        prices, volumes = sample_data
        features = analyzer.prepare_features(prices, volumes)
        features = features.unsqueeze(0)

        output = analyzer(features)

        assert output.shape == (1, 3)
        assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-5)

    def test_analyze(self, analyzer, sample_data):
        prices, volumes = sample_data
        result = analyzer.analyze(prices, volumes)

        assert "trend" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "indicators" in result

        assert isinstance(result["trend"], TrendDirection)
        assert 0 <= result["confidence"] <= 1
        assert set(result["probabilities"].keys()) == {"상승", "하락", "횡보"}
        assert set(result["indicators"].keys()) == {"rsi", "macd", "bollinger_position"}
