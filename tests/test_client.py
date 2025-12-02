"""CMCClient 테스트 (모킹)"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from crypto_ai.client import CMCClient
from crypto_ai.models import CryptoQuote


class TestCMCClient:
    """CMCClient 테스트"""

    @pytest.fixture
    def client(self):
        return CMCClient("test-api-key")

    @pytest.fixture
    def mock_quote_response(self):
        return {
            "data": {
                "BTC": [
                    {
                        "symbol": "BTC",
                        "name": "Bitcoin",
                        "quote": {
                            "USD": {
                                "price": 45000.0,
                                "volume_24h": 28000000000.0,
                                "market_cap": 880000000000.0,
                                "percent_change_1h": 0.5,
                                "percent_change_24h": -1.2,
                                "percent_change_7d": 5.3,
                                "last_updated": "2024-01-01T00:00:00.000Z",
                            }
                        },
                    }
                ]
            }
        }

    @pytest.fixture
    def mock_listings_response(self):
        return {
            "data": [
                {
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "quote": {
                        "USD": {
                            "price": 45000.0,
                            "volume_24h": 28000000000.0,
                            "market_cap": 880000000000.0,
                            "percent_change_1h": 0.5,
                            "percent_change_24h": -1.2,
                            "percent_change_7d": 5.3,
                            "last_updated": "2024-01-01T00:00:00.000Z",
                        }
                    },
                },
                {
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "quote": {
                        "USD": {
                            "price": 2500.0,
                            "volume_24h": 15000000000.0,
                            "market_cap": 300000000000.0,
                            "percent_change_1h": 0.3,
                            "percent_change_24h": 2.1,
                            "percent_change_7d": -3.2,
                            "last_updated": "2024-01-01T00:00:00.000Z",
                        }
                    },
                },
            ]
        }

    def test_initialization(self, client):
        assert client.api_key == "test-api-key"
        assert "X-CMC_PRO_API_KEY" in client.headers

    @patch("crypto_ai.client.requests.get")
    def test_get_quote(self, mock_get, client, mock_quote_response):
        mock_response = MagicMock()
        mock_response.json.return_value = mock_quote_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        quote = client.get_quote("BTC")

        assert quote is not None
        assert isinstance(quote, CryptoQuote)
        assert quote.symbol == "BTC"
        assert quote.price == 45000.0
        assert quote.percent_change_24h == -1.2

    @patch("crypto_ai.client.requests.get")
    def test_get_listings_latest(self, mock_get, client, mock_listings_response):
        mock_response = MagicMock()
        mock_response.json.return_value = mock_listings_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        quotes = client.get_listings_latest(limit=2)

        assert len(quotes) == 2
        assert all(isinstance(q, CryptoQuote) for q in quotes)
        assert quotes[0].symbol == "BTC"
        assert quotes[1].symbol == "ETH"

    @patch("crypto_ai.client.requests.get")
    def test_get_quote_not_found(self, mock_get, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        quote = client.get_quote("INVALID")

        assert quote is None
