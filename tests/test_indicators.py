import pytest
import pandas as pd
import numpy as np
from data.indicators import TechnicalIndicators

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 50
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })

class TestTechnicalIndicators:
    def test_add_ma(self, sample_df):
        result = TechnicalIndicators.add_ma(sample_df, period=10)
        assert "ma_10" in result.columns
        assert result["ma_10"].iloc[9:].notna().all()

    def test_add_ema(self, sample_df):
        result = TechnicalIndicators.add_ema(sample_df, period=10)
        assert "ema_10" in result.columns
        assert result["ema_10"].iloc[-1] != 0

    def test_add_rsi(self, sample_df):
        result = TechnicalIndicators.add_rsi(sample_df, period=14)
        assert "rsi_14" in result.columns
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_add_macd(self, sample_df):
        result = TechnicalIndicators.add_macd(sample_df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_add_bollinger(self, sample_df):
        result = TechnicalIndicators.add_bollinger(sample_df, period=20, std=2.0)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns

    def test_add_atr(self, sample_df):
        result = TechnicalIndicators.add_atr(sample_df, period=14)
        assert "atr_14" in result.columns
        valid = result["atr_14"].dropna()
        assert (valid > 0).all()

    def test_add_vwap(self, sample_df):
        result = TechnicalIndicators.add_vwap(sample_df)
        assert "vwap" in result.columns

    def test_add_obv(self, sample_df):
        result = TechnicalIndicators.add_obv(sample_df)
        assert "obv" in result.columns

    def test_add_all(self, sample_df):
        result = TechnicalIndicators.add_all(sample_df)
        expected_cols = ["ma_5", "ma_20", "ema_5", "ema_20", "rsi_14", "macd", "bb_upper", "atr_14", "vwap", "obv"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
