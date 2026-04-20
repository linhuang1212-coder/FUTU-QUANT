from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalAssetType(Enum):
    STOCK = "STOCK"
    OPTION = "OPTION"


@dataclass
class Signal:
    symbol: str
    direction: SignalDirection
    strength: float
    strategy_name: str
    reason: str
    suggested_type: SignalAssetType = SignalAssetType.STOCK
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength,
            "strategy_name": self.strategy_name,
            "reason": self.reason,
            "suggested_type": self.suggested_type.value,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseStrategy(ABC):
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    @abstractmethod
    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        pass

    def on_tick(self, symbol: str, tick_data: dict) -> Optional[Signal]:
        return None

    def get_params(self) -> dict:
        return self.params.copy()

    def set_params(self, params: dict) -> None:
        self.params.update(params)
