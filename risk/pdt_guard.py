from datetime import datetime, timedelta


class PdtGuard:
    def __init__(self, max_day_trades: int = 3, rolling_window_days: int = 5):
        self.max_day_trades = max_day_trades
        self.rolling_window_days = rolling_window_days
        self._trades: list[dict] = []

    def _recent_trades(self) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=self.rolling_window_days)
        return [t for t in self._trades if t["timestamp"] > cutoff]

    def can_day_trade(self) -> bool:
        return len(self._recent_trades()) < self.max_day_trades

    def remaining_day_trades(self) -> int:
        return max(0, self.max_day_trades - len(self._recent_trades()))

    def should_warn(self) -> bool:
        return self.remaining_day_trades() == 1

    def record_day_trade(self, symbol: str) -> None:
        self._trades.append({
            "symbol": symbol,
            "timestamp": datetime.now(),
        })

    def cleanup_old_trades(self) -> None:
        self._trades = self._recent_trades()
