from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.trade_store import TradeStore


class PdtGuard:
    """Pattern Day Trade guard.

    When a ``TradeStore`` is provided, day-trade records are persisted to
    SQLite so they survive process restarts.  Without one the guard falls
    back to an in-memory list (useful for backtests and unit tests).
    """

    def __init__(
        self,
        max_day_trades: int = 3,
        rolling_window_days: int = 5,
        trade_store: "TradeStore | None" = None,
    ):
        self.max_day_trades = max_day_trades
        self.rolling_window_days = rolling_window_days
        self._store = trade_store
        self._trades: list[dict] = []  # in-memory fallback

    # ── Internal helpers ─────────────────────────────────────────

    def _recent_count(self) -> int:
        if self._store is not None:
            return self._store.count_recent_day_trades(self.rolling_window_days)
        cutoff = datetime.now() - timedelta(days=self.rolling_window_days)
        return sum(1 for t in self._trades if t["timestamp"] > cutoff)

    def _recent_trades(self) -> list[dict]:
        """Return recent trades (kept for test compatibility)."""
        if self._store is not None:
            return self._store.get_recent_day_trades(self.rolling_window_days)
        cutoff = datetime.now() - timedelta(days=self.rolling_window_days)
        return [t for t in self._trades if t["timestamp"] > cutoff]

    # ── Public API ───────────────────────────────────────────────

    def can_day_trade(self) -> bool:
        return self._recent_count() < self.max_day_trades

    def remaining_day_trades(self) -> int:
        return max(0, self.max_day_trades - self._recent_count())

    def should_warn(self) -> bool:
        return self.remaining_day_trades() == 1

    def record_day_trade(self, symbol: str) -> None:
        if self._store is not None:
            self._store.record_day_trade(symbol)
        else:
            self._trades.append({
                "symbol": symbol,
                "timestamp": datetime.now(),
            })

    def cleanup_old_trades(self) -> None:
        if self._store is not None:
            self._store.cleanup_old_day_trades(self.rolling_window_days)
        else:
            self._trades = self._recent_trades()
