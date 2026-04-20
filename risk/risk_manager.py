from datetime import datetime, timedelta
from typing import Optional
from strategy.base import Signal


class RiskManager:
    def __init__(self, config: dict, initial_capital: float):
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_loss_time: Optional[datetime] = None

    def check_trade_allowed(self, signal: Signal, current_positions: dict, price: float) -> tuple[bool, str]:
        max_daily = self.initial_capital * self.config["max_daily_loss_pct"]
        if self.daily_loss >= max_daily:
            return False, f"Daily loss limit reached: ${self.daily_loss:.2f} >= ${max_daily:.2f}"

        if self.consecutive_losses >= self.config["max_consecutive_losses"]:
            if self.last_loss_time:
                cooldown_end = self.last_loss_time + timedelta(minutes=self.config["cooldown_minutes"])
                if datetime.now() < cooldown_end:
                    return False, f"Cooldown active after {self.consecutive_losses} consecutive losses"

        total_position_value = sum(p.get("value", 0) for p in current_positions.values())
        max_total = self.initial_capital * self.config["max_total_position_pct"]
        if total_position_value >= max_total:
            return False, f"Total position limit reached: ${total_position_value:.2f} >= ${max_total:.2f}"

        symbol_value = current_positions.get(signal.symbol, {}).get("value", 0)
        max_single = self.initial_capital * self.config["max_position_pct"]
        if symbol_value >= max_single:
            return False, f"Single position limit for {signal.symbol}: ${symbol_value:.2f} >= ${max_single:.2f}"

        return True, "Trade allowed"

    def calculate_position_size(self, price: float, signal_strength: float) -> int:
        max_value = self.initial_capital * self.config["max_position_pct"]
        strength_factor = signal_strength / 100.0
        target_value = max_value * strength_factor
        size = int(target_value / price)
        return max(size, 0)

    def record_loss(self, amount: float) -> None:
        self.daily_loss += amount
        self.consecutive_losses += 1
        self.last_loss_time = datetime.now()
        self.current_capital -= amount

    def record_win(self, amount: float) -> None:
        self.consecutive_losses = 0
        self.current_capital += amount

    def reset_daily(self) -> None:
        self.daily_loss = 0.0
