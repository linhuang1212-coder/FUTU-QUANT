"""FUTU-QUANT Live Trading Runner

Runs Momentum strategy on US.TQQQ with optimized parameters.
Uses daily K-line data for signal generation.

Usage:
    python run_live.py              # REAL trading
    python run_live.py --dry-run    # Test without placing orders
    python run_live.py --once       # Evaluate once and exit (for cron/scheduler)

IMPORTANT: Ensure FutuOpenD is running and logged in before starting.
"""

import sys
import io
import os
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from strategy.momentum import MomentumStrategy
from strategy.base import SignalDirection
from data.indicators import TechnicalIndicators
from utils.logger import setup_logger
from utils.helpers import load_yaml, get_project_root

SYMBOL = "US.TQQQ"

OPTIMAL_PARAMS = {
    "fast_ma_period": 8,
    "slow_ma_period": 15,
    "rsi_period": 10,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "volume_ratio_threshold": 1.0,
    "cross_lookback": 3,
    "rsi_momentum_enabled": True,
    "ema_trend_enabled": True,
}


class LiveTrader:
    def __init__(self, config_path: str, dry_run: bool = False):
        self.root = get_project_root()
        self.config = load_yaml(str(self.root / config_path))
        self.dry_run = dry_run
        self.strategy = MomentumStrategy(params=OPTIMAL_PARAMS)

        log_name = "live-dry" if dry_run else "live"
        self.logger = setup_logger(log_name, str(self.root / "data_store" / "logs" / f"{log_name}.log"))

        self.trade_log_dir = self.root / "data_store" / "trades"
        self.trade_log_dir.mkdir(parents=True, exist_ok=True)

        self._quote_ctx = None
        self._trade_ctx = None
        self._position_qty = 0
        self._position_avg_price = 0.0

    def connect(self) -> bool:
        from futu import OpenQuoteContext, OpenSecTradeContext, TrdEnv

        try:
            self._quote_ctx = OpenQuoteContext(
                host=self.config["futu"]["host"],
                port=self.config["futu"]["port"],
            )
            self.logger.info("Quote context connected")
        except Exception as e:
            self.logger.error(f"Quote connection failed: {e}")
            return False

        if not self.dry_run:
            try:
                env = TrdEnv.REAL if self.config["futu"]["trade_env"] == "REAL" else TrdEnv.SIMULATE
                self._trade_ctx = OpenSecTradeContext(
                    host=self.config["futu"]["host"],
                    port=self.config["futu"]["port"],
                )
                self.logger.info(f"Trade context connected ({self.config['futu']['trade_env']})")
            except Exception as e:
                self.logger.error(f"Trade connection failed: {e}")
                return False

        return True

    def disconnect(self):
        if self._quote_ctx:
            self._quote_ctx.close()
        if self._trade_ctx:
            self._trade_ctx.close()
        self.logger.info("Disconnected")

    def sync_position(self):
        """Check if we already hold TQQQ from a previous session."""
        if self._trade_ctx is None:
            return

        from futu import RET_OK, TrdEnv
        env = TrdEnv.REAL if self.config["futu"]["trade_env"] == "REAL" else TrdEnv.SIMULATE
        ret, data = self._trade_ctx.position_list_query(trd_env=env)
        if ret == RET_OK and data is not None:
            for _, row in data.iterrows():
                if row["code"] == SYMBOL:
                    self._position_qty = int(row["qty"])
                    self._position_avg_price = float(row["cost_price"])
                    self.logger.info(
                        f"Existing position found: {SYMBOL} x{self._position_qty} "
                        f"@ ${self._position_avg_price:.2f}"
                    )
                    return
        self._position_qty = 0
        self._position_avg_price = 0.0
        self.logger.info("No existing position in TQQQ")

    def get_daily_kline(self, count: int = 50) -> pd.DataFrame | None:
        """Fetch recent daily K-line data."""
        from futu import RET_OK, KLType

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=count * 2)).strftime("%Y-%m-%d")

        ret, data, _ = self._quote_ctx.request_history_kline(
            SYMBOL, start=start_date, end=end_date,
            ktype=KLType.K_DAY, max_count=count,
        )
        if ret == RET_OK and data is not None and len(data) >= 20:
            return data
        self.logger.error(f"Failed to get daily kline: {data if ret != 0 else 'insufficient data'}")
        return None

    def get_current_price(self) -> float | None:
        from futu import RET_OK
        ret, data = self._quote_ctx.get_market_snapshot([SYMBOL])
        if ret == RET_OK and data is not None and len(data) > 0:
            return float(data.iloc[0]["last_price"])
        return None

    def evaluate_signal(self) -> dict | None:
        """Get daily data, compute indicators, evaluate strategy."""
        df = self.get_daily_kline(50)
        if df is None:
            return None

        df = TechnicalIndicators.add_ma(df, 8)
        df = TechnicalIndicators.add_ma(df, 15)
        df = TechnicalIndicators.add_ema(df, 8)
        df = TechnicalIndicators.add_ema(df, 15)
        df = TechnicalIndicators.add_rsi(df, 10)

        signal = self.strategy.on_bar(SYMBOL, df)

        last_bar = df.iloc[-1]
        status = {
            "time": datetime.now().isoformat(),
            "close": float(last_bar["close"]),
            "ma_8": float(last_bar.get("ma_8", 0)),
            "ma_15": float(last_bar.get("ma_15", 0)),
            "ema_8": float(last_bar.get("ema_8", 0)),
            "ema_15": float(last_bar.get("ema_15", 0)),
            "rsi_10": float(last_bar.get("rsi_10", 0)),
            "volume": float(last_bar.get("volume", 0)),
            "position_qty": self._position_qty,
            "signal": None,
        }

        if signal is not None:
            status["signal"] = {
                "direction": signal.direction.value,
                "strength": signal.strength,
                "reason": signal.reason,
            }

        self.logger.info(
            f"[EVAL] close=${status['close']:.2f} "
            f"MA8={status['ma_8']:.2f} MA15={status['ma_15']:.2f} "
            f"RSI={status['rsi_10']:.1f} "
            f"pos={self._position_qty} "
            f"signal={signal.direction.value if signal else 'NONE'}"
        )

        return status if signal else None

    def execute_buy(self, price: float) -> bool:
        capital = self.config["account"]["initial_capital"]
        qty = int(capital * 0.95 / price)

        if qty <= 0:
            self.logger.warning("Calculated quantity is 0, skipping")
            return False

        cost = qty * price
        self.logger.info(f"[BUY] {SYMBOL} x{qty} @ ${price:.2f} = ${cost:.2f}")

        if self.dry_run:
            self.logger.info("[DRY-RUN] Order not placed")
            self._position_qty = qty
            self._position_avg_price = price
            self._log_trade("BUY", qty, price, dry_run=True)
            return True

        from futu import RET_OK, TrdSide, OrderType as FutuOrderType, TrdEnv
        env = TrdEnv.REAL if self.config["futu"]["trade_env"] == "REAL" else TrdEnv.SIMULATE
        ret, data = self._trade_ctx.place_order(
            price=price, qty=qty, code=SYMBOL,
            trd_side=TrdSide.BUY,
            order_type=FutuOrderType.MARKET,
            trd_env=env,
        )
        if ret == RET_OK:
            self.logger.info(f"[BUY CONFIRMED] order_id={data.iloc[0]['order_id'] if len(data) > 0 else 'N/A'}")
            self._position_qty = qty
            self._position_avg_price = price
            self._log_trade("BUY", qty, price)
            return True
        else:
            self.logger.error(f"[BUY FAILED] {data}")
            self._log_trade("BUY_FAILED", qty, price, error=str(data))
            return False

    def execute_sell(self, price: float) -> bool:
        if self._position_qty <= 0:
            self.logger.warning("No position to sell")
            return False

        qty = self._position_qty
        pnl = (price - self._position_avg_price) * qty
        pnl_pct = (price / self._position_avg_price - 1) * 100 if self._position_avg_price > 0 else 0

        self.logger.info(
            f"[SELL] {SYMBOL} x{qty} @ ${price:.2f} "
            f"PnL=${pnl:+.2f} ({pnl_pct:+.2f}%)"
        )

        if self.dry_run:
            self.logger.info("[DRY-RUN] Order not placed")
            self._position_qty = 0
            self._position_avg_price = 0.0
            self._log_trade("SELL", qty, price, pnl=pnl, dry_run=True)
            return True

        from futu import RET_OK, TrdSide, OrderType as FutuOrderType, TrdEnv
        env = TrdEnv.REAL if self.config["futu"]["trade_env"] == "REAL" else TrdEnv.SIMULATE
        ret, data = self._trade_ctx.place_order(
            price=price, qty=qty, code=SYMBOL,
            trd_side=TrdSide.SELL,
            order_type=FutuOrderType.MARKET,
            trd_env=env,
        )
        if ret == RET_OK:
            self.logger.info(f"[SELL CONFIRMED] order_id={data.iloc[0]['order_id'] if len(data) > 0 else 'N/A'}")
            self._position_qty = 0
            self._position_avg_price = 0.0
            self._log_trade("SELL", qty, price, pnl=pnl)
            return True
        else:
            self.logger.error(f"[SELL FAILED] {data}")
            self._log_trade("SELL_FAILED", qty, price, error=str(data))
            return False

    def _log_trade(self, action: str, qty: int, price: float,
                   pnl: float = 0, dry_run: bool = False, error: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "symbol": SYMBOL,
            "qty": qty,
            "price": price,
            "pnl": round(pnl, 2),
            "dry_run": dry_run,
            "error": error,
        }
        log_file = self.trade_log_dir / f"trades_{datetime.now().strftime('%Y%m')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def run_once(self):
        """Evaluate current market state and act if needed."""
        self.sync_position()

        result = self.evaluate_signal()

        if result is None:
            self.logger.info("No signal. Holding current position." if self._position_qty > 0 else "No signal. Staying flat.")
            return

        signal = result["signal"]
        current_price = self.get_current_price()
        if current_price is None:
            self.logger.error("Cannot get current price, aborting")
            return

        if signal["direction"] == "BUY" and self._position_qty == 0:
            self.logger.info(f">>> BUY SIGNAL: {signal['reason']} (strength={signal['strength']:.1f})")
            self.execute_buy(current_price)

        elif signal["direction"] == "SELL" and self._position_qty > 0:
            self.logger.info(f">>> SELL SIGNAL: {signal['reason']} (strength={signal['strength']:.1f})")
            self.execute_sell(current_price)

        else:
            if signal["direction"] == "BUY" and self._position_qty > 0:
                self.logger.info("BUY signal but already in position, holding")
            elif signal["direction"] == "SELL" and self._position_qty == 0:
                self.logger.info("SELL signal but no position, ignoring")

    def run_loop(self, interval: int = 300):
        """Run continuously, evaluating every `interval` seconds."""
        self.sync_position()

        print(f"\n{'='*60}")
        print(f"FUTU-QUANT Live Trading")
        print(f"Mode:     {'DRY-RUN' if self.dry_run else '*** REAL MONEY ***'}")
        print(f"Strategy: Momentum (fast=8, slow=15, RSI=10)")
        print(f"Symbol:   {SYMBOL}")
        print(f"Capital:  ${self.config['account']['initial_capital']:,.0f}")
        print(f"Position: {'x' + str(self._position_qty) + ' @ $' + f'{self._position_avg_price:.2f}' if self._position_qty > 0 else 'FLAT'}")
        print(f"Interval: {interval}s")
        print(f"{'='*60}")
        print("Press Ctrl+C to stop.\n")

        last_eval_date = None

        while True:
            try:
                now = datetime.now()
                today_str = now.strftime("%Y-%m-%d")

                if today_str != last_eval_date:
                    self.logger.info(f"--- New evaluation for {today_str} ---")
                    self.run_once()
                    last_eval_date = today_str
                    self.logger.info(f"Next evaluation tomorrow. Monitoring position...")
                else:
                    if self._position_qty > 0:
                        price = self.get_current_price()
                        if price:
                            pnl = (price - self._position_avg_price) * self._position_qty
                            pnl_pct = (price / self._position_avg_price - 1) * 100
                            self.logger.info(
                                f"[MONITOR] price=${price:.2f} "
                                f"PnL=${pnl:+.2f} ({pnl_pct:+.2f}%) "
                                f"pos=x{self._position_qty}"
                            )

                            hard_stop = self.config["risk"].get("hard_stop_pct", 0.08)
                            if pnl_pct < -hard_stop * 100:
                                self.logger.warning(
                                    f"[HARD STOP] Loss {pnl_pct:.2f}% exceeds "
                                    f"{hard_stop*100:.0f}% limit. SELLING."
                                )
                                self.execute_sell(price)

                time.sleep(interval)

            except KeyboardInterrupt:
                self.logger.info("Shutdown requested")
                break
            except Exception as e:
                self.logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="FUTU-QUANT Live Trading")
    parser.add_argument("--dry-run", action="store_true", help="Test without real orders")
    parser.add_argument("--once", action="store_true", help="Evaluate once and exit")
    parser.add_argument("--interval", type=int, default=300, help="Monitoring interval in seconds")
    parser.add_argument("--config", default="config/live.yaml", help="Config file path")
    args = parser.parse_args()

    trader = LiveTrader(config_path=args.config, dry_run=args.dry_run)

    if not trader.connect():
        print("ERROR: Failed to connect. Is FutuOpenD running?")
        sys.exit(1)

    try:
        if args.once:
            trader.run_once()
        else:
            trader.run_loop(interval=args.interval)
    finally:
        trader.disconnect()


if __name__ == "__main__":
    main()
