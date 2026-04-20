from enum import Enum, auto
from typing import Callable, Any
from collections import defaultdict


class EventType(Enum):
    MARKET_DATA = auto()
    SIGNAL = auto()
    ORDER = auto()
    RISK_ALERT = auto()
    SYSTEM = auto()


class EventBus:
    def __init__(self):
        self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Callable[[Any], None]) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Any], None]) -> None:
        self._subscribers[event_type] = [
            h for h in self._subscribers[event_type] if h is not handler
        ]

    def publish(self, event_type: EventType, data: Any = None) -> None:
        for handler in self._subscribers[event_type]:
            handler(data)
