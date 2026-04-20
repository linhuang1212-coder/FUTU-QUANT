import pytest
from datetime import datetime, timedelta
from risk.pdt_guard import PdtGuard

class TestPdtGuard:
    def setup_method(self):
        self.guard = PdtGuard(max_day_trades=3, rolling_window_days=5)

    def test_initially_allowed(self):
        assert self.guard.can_day_trade() is True
        assert self.guard.remaining_day_trades() == 3

    def test_record_day_trade(self):
        self.guard.record_day_trade("US.TQQQ")
        assert self.guard.remaining_day_trades() == 2

    def test_block_after_max(self):
        self.guard.record_day_trade("US.TQQQ")
        self.guard.record_day_trade("US.SOXL")
        self.guard.record_day_trade("US.SPY")
        assert self.guard.can_day_trade() is False
        assert self.guard.remaining_day_trades() == 0

    def test_old_trades_expire(self):
        old_time = datetime.now() - timedelta(days=6)
        self.guard._trades.append({"symbol": "US.TQQQ", "timestamp": old_time})
        self.guard._trades.append({"symbol": "US.SOXL", "timestamp": old_time})
        self.guard._trades.append({"symbol": "US.SPY", "timestamp": old_time})
        assert self.guard.can_day_trade() is True
        assert self.guard.remaining_day_trades() == 3

    def test_warning_threshold(self):
        self.guard.record_day_trade("US.TQQQ")
        self.guard.record_day_trade("US.SOXL")
        assert self.guard.should_warn() is True

    def test_no_warning_when_plenty_left(self):
        self.guard.record_day_trade("US.TQQQ")
        assert self.guard.should_warn() is False
