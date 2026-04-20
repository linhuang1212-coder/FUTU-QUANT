import pytest
from risk.risk_manager import RiskManager
from strategy.base import Signal, SignalDirection, SignalAssetType

def make_signal(symbol="US.TQQQ", direction=SignalDirection.BUY, strength=70):
    return Signal(
        symbol=symbol,
        direction=direction,
        strength=strength,
        strategy_name="test",
        reason="test signal",
    )

class TestRiskManager:
    def setup_method(self):
        self.config = {
            "max_loss_per_trade_pct": 0.05,
            "max_daily_loss_pct": 0.08,
            "max_position_pct": 0.40,
            "max_total_position_pct": 0.80,
            "max_consecutive_losses": 3,
            "cooldown_minutes": 60,
        }
        self.rm = RiskManager(self.config, initial_capital=3000)

    def test_calculate_position_size(self):
        size = self.rm.calculate_position_size(price=55.0, signal_strength=70)
        assert size > 0
        max_value = 3000 * 0.40
        assert size * 55.0 <= max_value

    def test_check_single_trade_loss(self):
        signal = make_signal()
        allowed, reason = self.rm.check_trade_allowed(signal, current_positions={}, price=55.0)
        assert allowed is True

    def test_reject_when_total_position_exceeded(self):
        positions = {
            "US.TQQQ": {"value": 1200},
            "US.SOXL": {"value": 700},
            "US.SPY": {"value": 600},
        }
        signal = make_signal(symbol="US.QQQ")
        allowed, reason = self.rm.check_trade_allowed(signal, current_positions=positions, price=450.0)
        assert allowed is False
        assert "total" in reason.lower() or "position" in reason.lower()

    def test_reject_after_daily_loss_exceeded(self):
        self.rm.record_loss(250)
        signal = make_signal()
        allowed, reason = self.rm.check_trade_allowed(signal, current_positions={}, price=55.0)
        assert allowed is False
        assert "daily" in reason.lower()

    def test_reject_after_consecutive_losses(self):
        for _ in range(3):
            self.rm.record_loss(50)
        signal = make_signal()
        allowed, reason = self.rm.check_trade_allowed(signal, current_positions={}, price=55.0)
        assert allowed is False
        assert "consecutive" in reason.lower() or "cooldown" in reason.lower()

    def test_record_win_resets_consecutive_losses(self):
        self.rm.record_loss(50)
        self.rm.record_loss(50)
        self.rm.record_win(100)
        assert self.rm.consecutive_losses == 0

    def test_reset_daily(self):
        self.rm.record_loss(200)
        self.rm.reset_daily()
        assert self.rm.daily_loss == 0
