import pytest
import pandas as pd
import numpy as np
from data.indicators import TechnicalIndicators
from strategy.base import BaseStrategy, Signal, SignalDirection
from strategy.multi_timeframe import MultiTimeframeStrategy


class AlwaysBuyStrategy(BaseStrategy):
    def __init__(self, params=None):
        super().__init__("always_buy", params or {})
        self._bought = False

    def on_bar(self, symbol, bar_data):
        if len(bar_data) < 5:
            return None
        if not self._bought:
            self._bought = True
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=70,
                strategy_name=self.name,
                reason="test buy",
            )
        return None


@pytest.fixture
def daily_data():
    np.random.seed(123)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 50 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "time_key": dates,
        "open": close - 0.3,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestResampleKline:
    def test_resample_daily_to_weekly(self, daily_data):
        weekly = TechnicalIndicators.resample_kline(daily_data, "1W")
        assert len(weekly) < len(daily_data)
        assert "close" in weekly.columns
        assert "volume" in weekly.columns
        assert len(weekly) > 0

    def test_resample_daily_to_monthly(self, daily_data):
        monthly = TechnicalIndicators.resample_kline(daily_data, "1ME")
        assert len(monthly) < len(daily_data)
        assert len(monthly) > 0

    def test_resample_preserves_ohlcv(self, daily_data):
        weekly = TechnicalIndicators.resample_kline(daily_data, "1W")
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in weekly.columns

    def test_resample_volume_sums(self, daily_data):
        weekly = TechnicalIndicators.resample_kline(daily_data, "1W")
        total_orig = daily_data["volume"].sum()
        total_resampled = weekly["volume"].sum()
        assert abs(total_orig - total_resampled) / total_orig < 0.01


class TestMultiTimeframeStrategy:
    def test_mtf_passes_signal_when_aligned(self, daily_data):
        inner = AlwaysBuyStrategy()
        mtf = MultiTimeframeStrategy(
            inner, htf_rule="1W", htf_ema_period=4, require_alignment=True
        )
        signal = mtf.on_bar("US.TEST", daily_data)
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_mtf_passthrough_when_disabled(self, daily_data):
        inner = AlwaysBuyStrategy()
        mtf = MultiTimeframeStrategy(
            inner, htf_rule="1W", htf_ema_period=4, require_alignment=False
        )
        signal = mtf.on_bar("US.TEST", daily_data)
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_mtf_reduces_contra_trend(self, daily_data):
        """When inner signal contradicts HTF trend, strength should be reduced."""
        inner = AlwaysBuyStrategy()
        mtf_aligned = MultiTimeframeStrategy(
            inner, htf_rule="1W", htf_ema_period=4, require_alignment=True
        )
        sig_aligned = mtf_aligned.on_bar("US.TEST", daily_data)

        inner2 = AlwaysBuyStrategy()
        downtrend_data = daily_data.copy()
        downtrend_data["close"] = np.linspace(100, 30, len(daily_data))
        downtrend_data["open"] = downtrend_data["close"] + 0.5
        downtrend_data["high"] = downtrend_data["close"] + 1.5
        downtrend_data["low"] = downtrend_data["close"] - 0.5

        mtf_contra = MultiTimeframeStrategy(
            inner2, htf_rule="1W", htf_ema_period=4, require_alignment=True
        )
        sig_contra = mtf_contra.on_bar("US.TEST", downtrend_data)

        assert sig_contra is not None
        if sig_aligned is not None:
            assert sig_contra.strength <= sig_aligned.strength

    def test_mtf_name_includes_inner(self):
        inner = AlwaysBuyStrategy()
        mtf = MultiTimeframeStrategy(inner)
        assert "always_buy" in mtf.name
