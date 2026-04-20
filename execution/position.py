from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("position")


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    strategy_name: str
    is_day_trade: bool = False
    open_time: datetime = field(default_factory=datetime.now)
    highest_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price

    def update_highest(self, current_price: float) -> None:
        if current_price > self.highest_price:
            self.highest_price = current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.avg_price == 0:
            return 0.0
        return (current_price - self.avg_price) / self.avg_price * 100


class PositionManager:
    def __init__(self):
        self._positions: dict[str, Position] = {}

    def open_position(self, symbol: str, quantity: int, price: float, strategy_name: str, is_day_trade: bool = False) -> Position:
        pos = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=price,
            strategy_name=strategy_name,
            is_day_trade=is_day_trade,
            highest_price=price,
        )
        self._positions[symbol] = pos
        logger.info(f"Opened position: {symbol} x{quantity} @ ${price:.2f}")
        return pos

    def close_position(self, symbol: str) -> Optional[Position]:
        pos = self._positions.pop(symbol, None)
        if pos:
            logger.info(f"Closed position: {symbol}")
        return pos

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        return self._positions.copy()

    def get_positions_dict(self) -> dict[str, dict]:
        return {
            sym: {"value": pos.market_value, "quantity": pos.quantity}
            for sym, pos in self._positions.items()
        }

    def get_day_trade_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.is_day_trade]

    def total_position_value(self) -> float:
        return sum(p.market_value for p in self._positions.values())

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions
