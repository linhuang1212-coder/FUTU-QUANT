"""Simulated-account validation runner.

Loads the top strategy-symbol combos from a parameter scan CSV,
then runs each as an independent $3,000 portfolio on the Futu
simulated (paper) account.

Usage:
    python run_simulation.py                         # Use scan results
    python run_simulation.py --scan results/scan_results.csv --top 5
    python run_simulation.py --dry-run               # No FutuOpenD needed
"""

import sys
import io
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from core.event_bus import EventBus, EventType
from core.scheduler import TradingScheduler
from data.market_data import MarketData
from data.indicators import TechnicalIndicators
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.breakout import BreakoutStrategy
from strategy.rsi_reversal import RsiReversalStrategy
from strategy.base import BaseStrategy, SignalDirection
from execution.trader import Trader
from execution.position import PositionManager
from risk.risk_manager import RiskManager
from risk.pdt_guard import PdtGuard
from utils.logger import setup_logger
from utils.helpers import load_yaml, get_project_root

logger = setup_logger("simulation")

STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "rsi_reversal": RsiReversalStrategy,
}


class SimulationSlot:
    """One independent $3,000 portfolio running a single strategy on a single symbol."""

    def __init__(
        self,
        slot_id: int,
        strategy_name: str,
        symbol: str,
        params: dict,
        capital: float = 3000,
        config: dict = None,
    ):
        self.slot_id = slot_id
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.capital = capital
        self.config = config or {}

        strat_cls = STRATEGY_MAP.get(strategy_name)
        if strat_cls is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        clean_params = {}
        for k, v in params.items():
            if hasattr(v, "item"):
                clean_params[k] = v.item()
            else:
                clean_params[k] = v
        self.strategy: BaseStrategy = strat_cls(params=clean_params)

        risk_cfg = self.config.get("risk", {
            "max_loss_per_trade_pct": 0.05,
            "max_daily_loss_pct": 0.08,
            "max_position_pct": 0.40,
            "max_total_position_pct": 0.80,
            "max_consecutive_losses": 3,
            "cooldown_minutes": 60,
        })

        self.event_bus = EventBus()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(config=risk_cfg, initial_capital=capital)
        self.pdt_guard = PdtGuard(
            max_day_trades=self.config.get("pdt", {}).get("max_day_trades", 3),
            rolling_window_days=self.config.get("pdt", {}).get("rolling_window_days", 5),
        )
        self.trader = Trader(
            risk_manager=self.risk_manager,
            pdt_guard=self.pdt_guard,
            position_manager=self.position_manager,
            event_bus=self.event_bus,
            trade_env=self.config.get("futu", {}).get("trade_env", "SIMULATE"),
            host=self.config.get("futu", {}).get("host", "127.0.0.1"),
            port=self.config.get("futu", {}).get("port", 11111),
        )

        self.trades: list[dict] = []
        self.signals_generated = 0
        self.signals_executed = 0

        def _on_order(data):
            self.trades.append({**data, "slot_id": self.slot_id, "time": datetime.now().isoformat()})
            self.signals_executed += 1

        self.event_bus.subscribe(EventType.ORDER, _on_order)

    def connect(self) -> bool:
        return self.trader.connect()

    def disconnect(self):
        self.trader.disconnect()

    def process_bar(self, kline: pd.DataFrame) -> None:
        """Evaluate strategy on latest bar data and execute if signaled."""
        signal = self.strategy.on_bar(self.symbol, kline)
        if signal is None:
            return

        self.signals_generated += 1
        current_price = float(kline.iloc[-1]["close"])
        logger.info(
            f"[Slot {self.slot_id}] {self.strategy_name}@{self.symbol}: "
            f"{signal.direction.value} strength={signal.strength:.1f} "
            f"price=${current_price:.2f}"
        )
        self.trader.execute_signal(signal, current_price, is_day_trade=True)

    def status(self) -> dict:
        positions = self.position_manager.get_all_positions()
        pos_value = self.position_manager.total_position_value()
        return {
            "slot_id": self.slot_id,
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "signals_generated": self.signals_generated,
            "signals_executed": self.signals_executed,
            "total_trades": len(self.trades),
            "open_positions": len(positions),
            "position_value": pos_value,
        }


def load_top_combos(scan_csv: str, top_n: int = 5) -> list[dict]:
    """Load top N combos from scan results CSV."""
    if not Path(scan_csv).exists():
        print(f"Scan results not found at {scan_csv}")
        print("Run `python run_param_scan.py` first.")
        return []

    df = pd.read_csv(scan_csv)
    if df.empty:
        return []

    metric_cols = {
        "sharpe_ratio", "total_return_pct", "max_drawdown_pct", "win_rate_pct",
        "total_trades", "profit_factor", "calmar_ratio", "overfit_score",
        "initial_capital", "final_capital", "total_return", "total_commission",
        "sortino_ratio", "cagr_pct", "exposure_pct", "avg_holding_period_days",
        "max_consecutive_losses", "max_consecutive_wins", "avg_win", "avg_loss",
        "monthly_returns", "benchmark_final_capital", "benchmark_total_return_pct",
        "strategy_vs_benchmark_pct", "train_rank",
    }

    combos = []
    for i, row in df.head(top_n).iterrows():
        strat = row.get("strategy", "")
        sym = row.get("symbol", "")
        params = {k: v for k, v in row.items()
                  if k not in metric_cols and k not in ("strategy", "symbol") and pd.notna(v)}
        combos.append({"strategy": strat, "symbol": sym, "params": params})
        print(f"  Combo #{len(combos)}: {strat} @ {sym} params={params}")

    return combos


def run_simulation(scan_csv: str, top_n: int, dry_run: bool, interval: int):
    root = get_project_root()
    config = load_yaml(str(root / "config" / "settings.yaml"))

    print("=" * 60)
    print("FUTU-QUANT Simulation Runner")
    print(f"Mode: {'DRY-RUN' if dry_run else 'SIMULATE'}")
    print(f"Capital per slot: $3,000")
    print("=" * 60)

    print("\n[1/3] Loading top combos from scan results...")
    combos = load_top_combos(scan_csv, top_n)
    if not combos:
        print("No combos to run. Exiting.")
        return

    if dry_run:
        config.setdefault("futu", {})["trade_env"] = "SIMULATE"

    print(f"\n[2/3] Initializing {len(combos)} simulation slots...")
    slots: list[SimulationSlot] = []
    for i, combo in enumerate(combos):
        slot = SimulationSlot(
            slot_id=i + 1,
            strategy_name=combo["strategy"],
            symbol=combo["symbol"],
            params=combo["params"],
            capital=3000,
            config=config,
        )
        if not dry_run:
            slot.connect()
        slots.append(slot)
        print(f"  Slot #{i+1}: {combo['strategy']} @ {combo['symbol']}")

    scheduler = TradingScheduler(
        timezone=config.get("scheduler", {}).get("timezone", "US/Eastern"),
        market_open=config.get("scheduler", {}).get("market_open", "09:30"),
        market_close=config.get("scheduler", {}).get("market_close", "16:00"),
    )

    md = None
    if not dry_run:
        md = MarketData(
            host=config.get("futu", {}).get("host", "127.0.0.1"),
            port=config.get("futu", {}).get("port", 11111),
        )
        if not md.connect():
            print("WARNING: Could not connect to FutuOpenD. Running in dry-run mode.")
            md = None

    symbols = list(set(c["symbol"] for c in combos))
    if md:
        md.subscribe(symbols)

    print(f"\n[3/3] Starting simulation loop (interval={interval}s)...")
    print("Press Ctrl+C to stop.\n")

    log_dir = Path("results/simulation_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sim_{session_id}.jsonl"

    iteration = 0
    try:
        while True:
            iteration += 1

            if not dry_run and not scheduler.is_market_hours():
                mins_info = f"Market closed. Next check in 30s."
                if iteration % 10 == 1:
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {mins_info}")
                time.sleep(30)
                continue

            for sym in symbols:
                kline = None
                if md:
                    kline = md.get_kline(sym, "K_1M", 100)
                    if kline is not None:
                        import pandas as _pd
                        if not isinstance(kline, _pd.DataFrame):
                            kline = _pd.DataFrame(kline)
                        kline = TechnicalIndicators.add_all(kline)

                if kline is None or len(kline) < 30:
                    continue

                for slot in slots:
                    if slot.symbol == sym:
                        slot.process_bar(kline)

            if iteration % 10 == 0:
                print(f"\n  --- Iteration {iteration} [{datetime.now().strftime('%H:%M:%S')}] ---")
                for slot in slots:
                    st = slot.status()
                    print(f"    Slot #{st['slot_id']}: "
                          f"signals={st['signals_generated']}, "
                          f"executed={st['signals_executed']}, "
                          f"trades={st['total_trades']}, "
                          f"positions={st['open_positions']}")

                    with open(log_file, "a", encoding="utf-8") as f:
                        entry = {**st, "timestamp": datetime.now().isoformat(), "iteration": iteration}
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nShutdown requested...")

    finally:
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        for slot in slots:
            st = slot.status()
            print(f"\n  Slot #{st['slot_id']}: {st['strategy']} @ {st['symbol']}")
            print(f"    Signals generated: {st['signals_generated']}")
            print(f"    Signals executed:  {st['signals_executed']}")
            print(f"    Total trades:      {st['total_trades']}")
            print(f"    Open positions:    {st['open_positions']}")
            slot.disconnect()

        if md:
            md.disconnect()

        summary = [s.status() for s in slots]
        summary_file = log_dir / f"sim_{session_id}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Session log: {log_file}")
        print(f"  Summary: {summary_file}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUTU-QUANT Simulation Runner")
    parser.add_argument("--scan", default="results/scan_results.csv", help="Path to scan results CSV")
    parser.add_argument("--top", type=int, default=5, help="Number of top combos to run")
    parser.add_argument("--dry-run", action="store_true", help="Run without FutuOpenD connection")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between iterations")
    args = parser.parse_args()
    run_simulation(args.scan, args.top, args.dry_run, args.interval)
