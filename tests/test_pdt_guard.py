import os
import tempfile
import pytest
from datetime import datetime, timedelta
from risk.pdt_guard import PdtGuard
from data.trade_store import TradeStore


class TestPdtGuardInMemory:
    """Original in-memory mode (no TradeStore)."""

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


class TestPdtGuardSQLite:
    """SQLite-backed mode via TradeStore."""

    def setup_method(self):
        self._tmpfile = tempfile.NamedTemporaryFile(
            suffix=".db", delete=False
        )
        self._tmpfile.close()
        self.store = TradeStore(db_path=self._tmpfile.name)
        self.guard = PdtGuard(
            max_day_trades=3,
            rolling_window_days=5,
            trade_store=self.store,
        )

    def teardown_method(self):
        self.store.close()
        os.unlink(self._tmpfile.name)

    def test_initially_allowed(self):
        assert self.guard.can_day_trade() is True
        assert self.guard.remaining_day_trades() == 3

    def test_record_persists(self):
        self.guard.record_day_trade("US.TQQQ")
        assert self.guard.remaining_day_trades() == 2

        # Simulate process restart: new PdtGuard, same DB
        guard2 = PdtGuard(
            max_day_trades=3,
            rolling_window_days=5,
            trade_store=self.store,
        )
        assert guard2.remaining_day_trades() == 2

    def test_block_after_max(self):
        self.guard.record_day_trade("US.TQQQ")
        self.guard.record_day_trade("US.SOXL")
        self.guard.record_day_trade("US.SPY")
        assert self.guard.can_day_trade() is False
        assert self.guard.remaining_day_trades() == 0

    def test_warning_threshold(self):
        self.guard.record_day_trade("US.TQQQ")
        self.guard.record_day_trade("US.SOXL")
        assert self.guard.should_warn() is True

    def test_cleanup_old(self):
        self.guard.record_day_trade("US.TQQQ")
        self.guard.record_day_trade("US.SOXL")
        self.guard.record_day_trade("US.SPY")
        assert self.guard.remaining_day_trades() == 0

        self.guard.cleanup_old_trades()
        # Records are recent, so they should survive cleanup
        assert self.guard.remaining_day_trades() == 0


class TestTradeStore:
    """Direct TradeStore tests."""

    def setup_method(self):
        self._tmpfile = tempfile.NamedTemporaryFile(
            suffix=".db", delete=False
        )
        self._tmpfile.close()
        self.store = TradeStore(db_path=self._tmpfile.name)

    def teardown_method(self):
        self.store.close()
        os.unlink(self._tmpfile.name)

    def test_log_and_query(self):
        self.store.log_trade(
            action="BUY", symbol="US.TQQQ", qty=10, price=50.0,
            strategy="momentum",
        )
        self.store.log_trade(
            action="SELL", symbol="US.TQQQ", qty=10, price=55.0,
            pnl=50.0, strategy="momentum",
        )
        trades = self.store.query_trades(symbol="US.TQQQ")
        assert len(trades) == 2
        assert trades[0]["action"] == "SELL"  # most recent first
        assert trades[1]["action"] == "BUY"

    def test_query_by_action(self):
        self.store.log_trade(action="BUY", symbol="US.TQQQ", qty=5, price=40.0)
        self.store.log_trade(action="SELL", symbol="US.TQQQ", qty=5, price=45.0)
        buys = self.store.query_trades(action="BUY")
        assert len(buys) == 1
        assert buys[0]["action"] == "BUY"

    def test_pdt_day_trade_count(self):
        self.store.record_day_trade("US.TQQQ")
        self.store.record_day_trade("US.SOXL")
        assert self.store.count_recent_day_trades(5) == 2

    def test_cleanup_preserves_recent(self):
        self.store.record_day_trade("US.TQQQ")
        deleted = self.store.cleanup_old_day_trades(5)
        assert deleted == 0
        assert self.store.count_recent_day_trades(5) == 1

    def test_dry_run_flag(self):
        self.store.log_trade(
            action="BUY", symbol="US.SPY", qty=1, price=100.0, dry_run=True,
        )
        trades = self.store.query_trades()
        assert trades[0]["dry_run"] == 1
