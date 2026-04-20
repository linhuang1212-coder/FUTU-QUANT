import time
from typing import Optional
from core.event_bus import EventBus, EventType
from core.scheduler import TradingScheduler
from data.market_data import MarketData
from data.history import HistoryManager
from data.indicators import TechnicalIndicators
from strategy.base import BaseStrategy, SignalDirection
from execution.trader import Trader
from execution.position import PositionManager
from risk.risk_manager import RiskManager
from risk.pdt_guard import PdtGuard
from notification.telegram_bot import TelegramNotifier
from utils.logger import setup_logger
from utils.helpers import load_yaml, get_project_root

logger = setup_logger("engine")


class Engine:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.root = get_project_root()
        self.config = load_yaml(str(self.root / config_path))
        self.strategies_config = load_yaml(str(self.root / "config" / "strategies.yaml"))
        self.symbols_config = load_yaml(str(self.root / "config" / "symbols.yaml"))

        self.event_bus = EventBus()
        self.scheduler = TradingScheduler(
            timezone=self.config["scheduler"]["timezone"],
            market_open=self.config["scheduler"]["market_open"],
            market_close=self.config["scheduler"]["market_close"],
        )
        self.market_data = MarketData(
            host=self.config["futu"]["host"],
            port=self.config["futu"]["port"],
        )
        self.history = HistoryManager()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(
            config=self.config["risk"],
            initial_capital=self.config["account"]["initial_capital"],
        )
        self.pdt_guard = PdtGuard(
            max_day_trades=self.config["pdt"]["max_day_trades"],
            rolling_window_days=self.config["pdt"]["rolling_window_days"],
        )
        self.trader = Trader(
            risk_manager=self.risk_manager,
            pdt_guard=self.pdt_guard,
            position_manager=self.position_manager,
            event_bus=self.event_bus,
            trade_env=self.config["futu"]["trade_env"],
            host=self.config["futu"]["host"],
            port=self.config["futu"]["port"],
        )

        tg_config = self.config.get("telegram", {})
        self.notifier = TelegramNotifier(
            bot_token=tg_config.get("bot_token", ""),
            chat_id=tg_config.get("chat_id", ""),
            enabled=tg_config.get("enabled", False),
        )

        self.strategies: list[BaseStrategy] = []
        self._running = False

    def load_strategies(self) -> None:
        from strategy.momentum import MomentumStrategy
        from strategy.mean_reversion import MeanReversionStrategy
        from strategy.breakout import BreakoutStrategy
        from strategy.rsi_reversal import RsiReversalStrategy

        strategy_map = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy,
            "rsi_reversal": RsiReversalStrategy,
        }
        for name, cfg in self.strategies_config.get("strategies", {}).items():
            if cfg.get("enabled", False) and name in strategy_map:
                strategy = strategy_map[name](params=cfg.get("params", {}))
                self.strategies.append(strategy)
                logger.info(f"Loaded strategy: {name}")

    def get_symbols(self) -> list[str]:
        symbols = []
        for category in ["leveraged", "standard"]:
            symbols.extend(self.symbols_config.get("etf", {}).get(category, []))
        symbols.extend(self.symbols_config.get("stocks", []))
        return symbols

    def _setup_event_handlers(self) -> None:
        def on_order(data):
            if data.get("direction") == "BUY":
                self.notifier.notify_open_position(
                    data["symbol"], data["quantity"], data["price"],
                    data["strategy_name"], data.get("strength", 0)
                )
            else:
                self.notifier.notify_close_position(
                    data["symbol"], data["quantity"], data["price"],
                    data.get("pnl", 0), data.get("pnl_pct", 0)
                )

        def on_risk_alert(data):
            reason = data.get("reason", "")
            if "PDT" in reason:
                self.notifier.notify_pdt_warning(self.pdt_guard.remaining_day_trades())

        self.event_bus.subscribe(EventType.ORDER, on_order)
        self.event_bus.subscribe(EventType.RISK_ALERT, on_risk_alert)

    def start(self) -> None:
        logger.info("FUTU-QUANT Engine starting...")
        self._setup_event_handlers()
        self.load_strategies()

        connected = self.market_data.connect()
        if not connected:
            logger.warning("FutuOpenD not available, running in dry-run mode")

        self.trader.connect()
        self.notifier.notify_system_start(self.config["account"]["initial_capital"])

        symbols = self.get_symbols()
        if connected:
            self.market_data.subscribe(symbols)

        self._running = True
        logger.info(f"Engine started with {len(self.strategies)} strategies, {len(symbols)} symbols")

        try:
            self._run_loop(symbols)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        finally:
            self.stop()

    def _run_loop(self, symbols: list[str]) -> None:
        interval = self.config["scheduler"].get("bar_interval_seconds", 60)
        eod_minutes = self.config["risk"].get("eod_close_minutes_before", 15)

        while self._running:
            if not self.scheduler.is_market_hours():
                logger.debug("Market closed, waiting...")
                time.sleep(30)
                continue

            if self.scheduler.should_force_close_day_trades(eod_minutes):
                self._force_close_day_trades()

            for symbol in symbols:
                self._process_symbol(symbol)

            time.sleep(interval)

    def _process_symbol(self, symbol: str) -> None:
        kline = self.market_data.get_kline(symbol, "K_1M", 100)
        if kline is None:
            return

        import pandas as pd
        if not isinstance(kline, pd.DataFrame):
            kline = pd.DataFrame(kline)

        kline = TechnicalIndicators.add_all(kline)

        for strategy in self.strategies:
            signal = strategy.on_bar(symbol, kline)
            if signal is not None:
                logger.info(f"Signal: {signal.direction.value} {signal.symbol} strength={signal.strength} from {signal.strategy_name}")
                self.event_bus.publish(EventType.SIGNAL, signal.to_dict())

                current_price = kline.iloc[-1]["close"]
                is_day = not self.scheduler.should_force_close_day_trades()
                self.trader.execute_signal(signal, current_price, is_day_trade=is_day)

    def _force_close_day_trades(self) -> None:
        day_positions = self.position_manager.get_day_trade_positions()
        for pos in day_positions:
            logger.info(f"Force closing day trade: {pos.symbol}")
            from strategy.base import Signal, SignalDirection
            close_signal = Signal(
                symbol=pos.symbol,
                direction=SignalDirection.SELL,
                strength=100,
                strategy_name="eod_force_close",
                reason="End of day forced close",
            )
            self.trader.execute_signal(close_signal, pos.avg_price)

    def stop(self) -> None:
        self._running = False
        self.market_data.disconnect()
        self.trader.disconnect()
        self.risk_manager.reset_daily()
        logger.info("Engine stopped")
