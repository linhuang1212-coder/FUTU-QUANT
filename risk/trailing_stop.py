"""Tiered trailing stop with ATR-based dynamic adjustment.

Tracks highest price since entry. When profit exceeds tier thresholds,
activates trailing stops that lock in gains. ATR-based stop provides
a volatility-aware alternative.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrailingStopState:
    entry_price: float
    highest_price: float
    active_tier: int  # 0=inactive, 1=tier1, 2=tier2
    stop_price: float

    @property
    def pnl_from_peak(self) -> float:
        if self.highest_price <= 0:
            return 0.0
        return (self.highest_price - self.stop_price) / self.entry_price


class TrailingStopManager:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.activate_pct = config.get("activate_pct", 0.05)
        self.trail_pct = config.get("trail_pct", 0.03)
        self.tier2_activate_pct = config.get("tier2_activate_pct", 0.15)
        self.tier2_trail_pct = config.get("tier2_trail_pct", 0.02)
        self.atr_enabled = config.get("atr_enabled", True)
        self.atr_multiplier = config.get("atr_multiplier", 2.5)
        self._states: dict[str, TrailingStopState] = {}

    def on_entry(self, symbol: str, entry_price: float):
        self._states[symbol] = TrailingStopState(
            entry_price=entry_price,
            highest_price=entry_price,
            active_tier=0,
            stop_price=0.0,
        )

    def on_exit(self, symbol: str):
        self._states.pop(symbol, None)

    def update(
        self, symbol: str, current_price: float, current_atr: float = 0.0
    ) -> Optional[str]:
        """Update trailing stop state. Returns exit reason if stop triggered."""
        if not self.enabled or symbol not in self._states:
            return None

        state = self._states[symbol]

        if current_price > state.highest_price:
            state.highest_price = current_price

        pnl_pct = (state.highest_price / state.entry_price) - 1.0

        if pnl_pct >= self.tier2_activate_pct:
            state.active_tier = 2
            trail = self.tier2_trail_pct
        elif pnl_pct >= self.activate_pct:
            state.active_tier = 1
            trail = self.trail_pct
        else:
            state.active_tier = 0
            return None

        fixed_stop = state.highest_price * (1 - trail)

        if self.atr_enabled and current_atr > 0:
            atr_stop = state.highest_price - self.atr_multiplier * current_atr
            stop = max(fixed_stop, atr_stop)
        else:
            stop = fixed_stop

        state.stop_price = max(state.stop_price, stop)

        if current_price <= state.stop_price:
            pnl = (current_price / state.entry_price - 1) * 100
            return (
                f"Trailing stop T{state.active_tier} triggered: "
                f"peak=${state.highest_price:.2f} stop=${state.stop_price:.2f} "
                f"PnL={pnl:+.1f}%"
            )

        return None

    def get_state(self, symbol: str) -> Optional[TrailingStopState]:
        return self._states.get(symbol)
