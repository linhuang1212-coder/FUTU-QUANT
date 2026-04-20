import pytest
from execution.order import Order, OrderDirection, OrderType, OrderStatus

class TestOrder:
    def test_create_buy_order(self):
        order = Order(
            symbol="US.TQQQ",
            direction=OrderDirection.BUY,
            quantity=10,
            price=55.0,
            order_type=OrderType.LIMIT,
            strategy_name="momentum"
        )
        assert order.symbol == "US.TQQQ"
        assert order.direction == OrderDirection.BUY
        assert order.quantity == 10
        assert order.status == OrderStatus.PENDING
        assert order.strategy_name == "momentum"

    def test_order_fill(self):
        order = Order(
            symbol="US.SOXL",
            direction=OrderDirection.SELL,
            quantity=40,
            price=18.0,
            order_type=OrderType.MARKET,
            strategy_name="breakout"
        )
        order.fill(fill_price=18.1, fill_quantity=40)
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == 18.1

    def test_order_cancel(self):
        order = Order(
            symbol="US.SPY",
            direction=OrderDirection.BUY,
            quantity=2,
            price=530.0,
            order_type=OrderType.LIMIT,
            strategy_name="mean_reversion"
        )
        order.cancel()
        assert order.status == OrderStatus.CANCELLED

    def test_order_to_dict(self):
        order = Order(
            symbol="US.TQQQ",
            direction=OrderDirection.BUY,
            quantity=10,
            price=55.0,
            order_type=OrderType.LIMIT,
            strategy_name="momentum"
        )
        d = order.to_dict()
        assert d["symbol"] == "US.TQQQ"
        assert d["direction"] == "BUY"
        assert "timestamp" in d
