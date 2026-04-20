import pytest
import pandas as pd
import numpy as np
from backtest.optimizer import ParameterOptimizer
from strategy.base import BaseStrategy, Signal, SignalDirection


class SimpleTestStrategy(BaseStrategy):
    def __init__(self, params=None):
        default = {"threshold": 50}
        if params:
            default.update(params)
        super().__init__("test_strat", default)

    def on_bar(self, symbol, bar_data):
        if len(bar_data) < 5:
            return None
        curr = bar_data.iloc[-1]
        if curr["close"] < self.params["threshold"]:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=70,
                strategy_name=self.name,
                reason="test",
            )
        if curr["close"] > self.params["threshold"] + 5:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=70,
                strategy_name=self.name,
                reason="test",
            )
        return None


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    close = 50 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "time_key": pd.date_range("2025-01-01", periods=n, freq="D"),
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestParameterOptimizer:
    def test_grid_search(self, sample_data):
        opt = ParameterOptimizer(initial_capital=3000)
        results = opt.grid_search(
            SimpleTestStrategy,
            "US.TEST",
            sample_data,
            param_grid={"threshold": [48, 50, 52]},
            sort_by="total_return_pct",
            min_trades=0,
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert "threshold" in results.columns
        assert "total_return_pct" in results.columns

    def test_walk_forward(self, sample_data):
        opt = ParameterOptimizer(initial_capital=3000)
        result = opt.walk_forward(
            SimpleTestStrategy,
            "US.TEST",
            sample_data,
            param_grid={"threshold": [48, 50, 52]},
            train_pct=0.7,
            min_trades=0,
        )
        assert "train_results" in result
        assert "validation_results" in result
        assert "best_params" in result

    def test_multi_symbol_scan(self, sample_data):
        opt = ParameterOptimizer(initial_capital=3000)
        results = opt.multi_symbol_scan(
            SimpleTestStrategy,
            {"US.A": sample_data, "US.B": sample_data},
            param_grid={"threshold": [48, 52]},
            min_trades=0,
        )
        assert "symbol" in results.columns
