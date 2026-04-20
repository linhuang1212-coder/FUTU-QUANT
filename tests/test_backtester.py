import pytest
import pandas as pd
import numpy as np
from backtest.backtester import Backtester
from backtest.report import BacktestReport
from strategy.base import BaseStrategy, Signal, SignalDirection

class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("dummy", {})
        self._call_count = 0

    def on_bar(self, symbol, bar_data):
        self._call_count += 1
        if self._call_count % 10 == 0:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=70,
                strategy_name="dummy",
                reason="test buy"
            )
        if self._call_count % 15 == 0:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=70,
                strategy_name="dummy",
                reason="test sell"
            )
        return None

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    close = 50 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "time_key": pd.date_range("2025-01-01", periods=n, freq="D"),
        "open": close - np.random.rand(n) * 0.3,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })

class TestBacktester:
    def test_run_backtest(self, sample_data):
        bt = Backtester(initial_capital=3000, commission_pct=0.001)
        strategy = DummyStrategy()
        result = bt.run(strategy, "US.TQQQ", sample_data)
        assert "trades" in result
        assert "final_capital" in result
        assert result["final_capital"] > 0

    def test_backtest_report(self, sample_data):
        bt = Backtester(initial_capital=3000, commission_pct=0.001)
        strategy = DummyStrategy()
        result = bt.run(strategy, "US.TQQQ", sample_data)
        assert "data" in result
        report = BacktestReport(result)
        summary = report.summary()
        assert "total_return_pct" in summary
        assert "max_drawdown_pct" in summary
        assert "total_trades" in summary
        assert "sharpe_ratio" in summary
        assert "sortino_ratio" in summary
        assert "calmar_ratio" in summary
        assert "cagr_pct" in summary
        assert "exposure_pct" in summary
        assert "max_consecutive_losses" in summary
        assert "max_consecutive_wins" in summary
        assert "monthly_returns" in summary
        assert "profit_factor" in summary
        assert "avg_holding_period_days" in summary
        assert isinstance(summary["monthly_returns"], dict)
        assert summary["benchmark_final_capital"] > 0
        assert "benchmark_total_return_pct" in summary
        assert summary["max_consecutive_losses"] >= 0
        assert summary["max_consecutive_wins"] >= 0
        assert 0 <= summary["exposure_pct"] <= 100
        out = report.print_report()
        assert isinstance(out, str)
        assert "BACKTEST REPORT" in out
        assert "BUY & HOLD" in out
        assert "MONTHLY RETURNS" in out

    def test_backtest_report_skips_benchmark_without_data(self):
        result = {
            "initial_capital": 1000.0,
            "final_capital": 1050.0,
            "trades": [],
            "equity_curve": [1000.0, 1010.0, 1050.0],
            "total_bars": 3,
        }
        summary = BacktestReport(result).summary()
        assert "benchmark_final_capital" not in summary
        text = BacktestReport(result).print_report()
        assert "BUY & HOLD" not in text
