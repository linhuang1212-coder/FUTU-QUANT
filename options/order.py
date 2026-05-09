from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class OptionLeg:
    code: str               # Futu option code e.g. "US.SPY260501C550000"
    underlying: str         # "US.SPY"
    direction: str          # "BUY" / "SELL"
    qty: int                # contracts
    option_type: str        # "CALL" / "PUT"
    strike: float
    expiry: str             # "2026-05-01"
    price: float = 0.0      # limit price
    fill_price: float = 0.0
    order_id: str = ""

    def to_dict(self) -> dict:
        return {
            "code": self.code, "underlying": self.underlying,
            "direction": self.direction, "qty": self.qty,
            "option_type": self.option_type, "strike": self.strike,
            "expiry": self.expiry, "price": self.price,
            "fill_price": self.fill_price, "order_id": self.order_id,
        }

@dataclass
class OptionTrade:
    strategy: str           # "orb_0dte" / "earnings_spread" / "credit_spread" / "straddle"
    underlying: str
    legs: list[OptionLeg] = field(default_factory=list)
    max_loss: float = 0.0
    target_pnl: float = 0.0
    stop_loss_pct: float = 0.50
    take_profit_pct: float = 1.00
    status: str = "pending"  # pending / open / closed
    open_timestamp: Optional[str] = None
    close_timestamp: Optional[str] = None
    realized_pnl: float = 0.0
    close_reason: str = ""
    dry_run: bool = False
    trade_id: Optional[int] = None  # DB row id
    direction_details: str = ""     # Direction analysis summary

    def net_premium(self) -> float:
        """Net premium paid (positive) or received (negative)."""
        total = 0.0
        for leg in self.legs:
            p = leg.fill_price if leg.fill_price > 0 else leg.price
            if leg.direction == "BUY":
                total += p * leg.qty * 100
            else:
                total -= p * leg.qty * 100
        return total

    def legs_json(self) -> str:
        return json.dumps([l.to_dict() for l in self.legs])

    @staticmethod
    def legs_from_json(s: str) -> list[OptionLeg]:
        data = json.loads(s)
        return [OptionLeg(**d) for d in data]
