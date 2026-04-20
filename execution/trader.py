from typing import Optional
from execution.order import Order, OrderDirection, OrderType, OrderStatus
from execution.position import PositionManager
from strategy.base import Signal, SignalDirection
from risk.risk_manager import RiskManager
from risk.pdt_guard import PdtGuard
from core.event_bus import EventBus, EventType
from utils.logger import setup_logger

logger = setup_logger("trader")


class Trader:
    def __init__(
        self,
        risk_manager: RiskManager,
        pdt_guard: PdtGuard,
        position_manager: PositionManager,
        event_bus: EventBus,
        trade_env: str = "SIMULATE",
        host: str = "127.0.0.1",
        port: int = 11111,
    ):
        self.risk_manager = risk_manager
        self.pdt_guard = pdt_guard
        self.position_manager = position_manager
        self.event_bus = event_bus
        self.trade_env = trade_env
        self.host = host
        self.port = port
        self._trade_ctx = None
        self._order_history: list[Order] = []

    def connect(self) -> bool:
        try:
            from futu import OpenSecTradeContext, TrdEnv
            env = TrdEnv.SIMULATE if self.trade_env == "SIMULATE" else TrdEnv.REAL
            self._trade_ctx = OpenSecTradeContext(
                host=self.host, port=self.port, filter_trdmarket=None, security_firm=None
            )
            logger.info(f"Trader connected ({self.trade_env})")
            return True
        except Exception as e:
            logger.error(f"Trader connection failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._trade_ctx:
            self._trade_ctx.close()
            logger.info("Trader disconnected")

    def execute_signal(self, signal: Signal, price: float, is_day_trade: bool = False) -> Optional[Order]:
        positions = self.position_manager.get_positions_dict()
        allowed, reason = self.risk_manager.check_trade_allowed(signal, positions, price)
        if not allowed:
            logger.warning(f"Trade rejected: {reason}")
            self.event_bus.publish(EventType.RISK_ALERT, {"signal": signal.to_dict(), "reason": reason})
            return None

        if is_day_trade and not self.pdt_guard.can_day_trade():
            logger.warning("Trade rejected: PDT limit reached")
            self.event_bus.publish(EventType.RISK_ALERT, {"signal": signal.to_dict(), "reason": "PDT limit"})
            return None

        quantity = self.risk_manager.calculate_position_size(price, signal.strength)
        if quantity <= 0:
            logger.warning("Calculated position size is 0")
            return None

        direction = OrderDirection.BUY if signal.direction == SignalDirection.BUY else OrderDirection.SELL
        order = Order(
            symbol=signal.symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            order_type=OrderType.MARKET,
            strategy_name=signal.strategy_name,
        )

        success = self._place_order(order)
        if success:
            if direction == OrderDirection.BUY:
                self.position_manager.open_position(
                    signal.symbol, quantity, price, signal.strategy_name, is_day_trade
                )
                if is_day_trade:
                    self.pdt_guard.record_day_trade(signal.symbol)
            elif direction == OrderDirection.SELL:
                self.position_manager.close_position(signal.symbol)

            self._order_history.append(order)
            self.event_bus.publish(EventType.ORDER, order.to_dict())

        return order

    def _place_order(self, order: Order) -> bool:
        try:
            if self._trade_ctx is None:
                logger.info(f"[DRY RUN] {order.direction.value} {order.symbol} x{order.quantity} @ ${order.price:.2f}")
                order.fill(order.price, order.quantity)
                return True

            from futu import RET_OK, TrdSide, OrderType as FutuOrderType
            side = TrdSide.BUY if order.direction == OrderDirection.BUY else TrdSide.SELL
            ret, data = self._trade_ctx.place_order(
                price=order.price,
                qty=order.quantity,
                code=order.symbol,
                trd_side=side,
                order_type=FutuOrderType.MARKET,
            )
            if ret == RET_OK:
                order.fill(order.price, order.quantity)
                logger.info(f"Order filled: {order.symbol} {order.direction.value} x{order.quantity}")
                return True
            else:
                order.status = OrderStatus.FAILED
                logger.error(f"Order failed: {data}")
                return False
        except Exception as e:
            order.status = OrderStatus.FAILED
            logger.error(f"Order execution error: {e}")
            return False

    def get_order_history(self) -> list[Order]:
        return self._order_history.copy()
