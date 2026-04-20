import pytest
from core.event_bus import EventBus, EventType

class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.SIGNAL, lambda data: received.append(data))
        bus.publish(EventType.SIGNAL, {"direction": "BUY"})
        assert len(received) == 1
        assert received[0]["direction"] == "BUY"

    def test_multiple_subscribers(self):
        bus = EventBus()
        results_a = []
        results_b = []
        bus.subscribe(EventType.MARKET_DATA, lambda d: results_a.append(d))
        bus.subscribe(EventType.MARKET_DATA, lambda d: results_b.append(d))
        bus.publish(EventType.MARKET_DATA, {"price": 100})
        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        handler = lambda d: received.append(d)
        bus.subscribe(EventType.ORDER, handler)
        bus.unsubscribe(EventType.ORDER, handler)
        bus.publish(EventType.ORDER, {"id": 1})
        assert len(received) == 0

    def test_different_event_types_isolated(self):
        bus = EventBus()
        signal_data = []
        order_data = []
        bus.subscribe(EventType.SIGNAL, lambda d: signal_data.append(d))
        bus.subscribe(EventType.ORDER, lambda d: order_data.append(d))
        bus.publish(EventType.SIGNAL, {"x": 1})
        assert len(signal_data) == 1
        assert len(order_data) == 0

    def test_publish_with_no_subscribers(self):
        bus = EventBus()
        bus.publish(EventType.SYSTEM, {"status": "ok"})
