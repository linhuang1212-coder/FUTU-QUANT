from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional


class OrderDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Order:
    symbol: str
    direction: OrderDirection
    quantity: int
    price: float
    order_type: OrderType
    strategy_name: str
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_quantity: Optional[int] = None
    order_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    fill_timestamp: Optional[datetime] = None

    def fill(self, fill_price: float, fill_quantity: int) -> None:
        self.fill_price = fill_price
        self.fill_quantity = fill_quantity
        self.status = OrderStatus.FILLED if fill_quantity >= self.quantity else OrderStatus.PARTIAL
        self.fill_timestamp = datetime.now()

    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "strategy_name": self.strategy_name,
            "fill_price": self.fill_price,
            "fill_quantity": self.fill_quantity,
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat(),
            "fill_timestamp": self.fill_timestamp.isoformat() if self.fill_timestamp else None,
        }
