"""Parameter scan pipeline: find top strategy+param combos across symbols.

Usage:
    python run_param_scan.py
    python run_param_scan.py --output results/scan_results.csv

Connects to FutuOpenD to fetch historical data, runs walk-forward
optimization for every (strategy, symbol) pair, and writes a ranked CSV.
"""

import sys
import io
import argparse
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from backtest.optimizer import ParameterOptimizer
from data.market_data import MarketData
from data.history import HistoryManager
from data.indicators import TechnicalIndicators
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.breakout import BreakoutStrategy
from strategy.rsi_reversal import RsiReversalStrategy
from utils.helpers import load_yaml, get_project_root

SYMBOLS = ["US.TQQQ", "US.SOXL", "US.TNA", "US.QQQ", "US.SPY"]

STRATEGY_GRIDS = {
    "momentum": {
        "class": MomentumStrategy,
        "grid": {
            "fast_ma_period": [3, 5, 8],
            "slow_ma_period": [10, 15, 20],
            "rsi_period": [7, 10, 14],
            "volume_ratio_threshold": [1.0, 1.2, 1.5],
            "cross_lookback": [2, 3, 5],
        },
    },
    "mean_reversion": {
        "class": MeanReversionStrategy,
        "grid": {
            "bb_period": [10, 15, 20],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_period": [7, 10, 14],
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "use_or_logic": [True],
        },
    },
    "breakout": {
        "class": BreakoutStrategy,
        "grid": {
            "lookback_period": [10, 15, 20],
            "volume_weak_ratio": [1.2, 1.3, 1.5],
            "macd_required": [False],
            "atr_breakout_enabled": [True, False],
            "donchian_enabled": [True, False],
        },
    },
    "rsi_reversal": {
        "class": RsiReversalStrategy,
        "grid": {
            "rsi_period": [5, 7, 10],
            "rsi_buy_threshold": [25, 30, 35],
            "rsi_sell_threshold": [65, 70, 75],
            "use_volume_filter": [True, False],
        },
    },
}


def fetch_all_data(symbols: list[str], days: int = 365) -> dict[str, pd.DataFrame]:
    """Fetch historical daily K-line data for each symbol."""
    root = get_project_root()
    settings = load_yaml(str(root / "config" / "settings.yaml"))

    md = MarketData(host=settings["futu"]["host"], port=settings["futu"]["port"])
    hm = HistoryManager()
    result = {}

    connected = md.connect()
    if not connected:
        print("[WARN] FutuOpenD unavailable — trying cache only")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    for sym in symbols:
        df = hm.get_history(sym, "K_DAY", start_date, end_date, market_data=md if connected else None)
        if df is not None and len(df) > 50:
            df = TechnicalIndicators.add_all(df)
            result[sym] = df
            print(f"  {sym}: {len(df)} bars loaded")
        else:
            print(f"  {sym}: insufficient data, skipping")

    if connected:
        md.disconnect()

    return result


def run_scan(output_path: str = "results/scan_results.csv"):
    print("=" * 60)
    print("FUTU-QUANT Parameter Scan Pipeline")
    print("=" * 60)

    print("\n[1/3] Fetching historical data...")
    all_data = fetch_all_data(SYMBOLS)

    if not all_data:
        print("ERROR: No data available. Ensure FutuOpenD is running.")
        return

    optimizer = ParameterOptimizer(initial_capital=3000)
    all_results = []

    print(f"\n[2/3] Running walk-forward optimization...")
    for strat_name, cfg in STRATEGY_GRIDS.items():
        strat_cls = cfg["class"]
        grid = cfg["grid"]
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"\n  Strategy: {strat_name} ({n_combos} combos)")

        for sym, data in all_data.items():
            print(f"    Scanning {sym}...", end=" ")
            try:
                wf = optimizer.walk_forward(
                    strat_cls, sym, data, grid,
                    train_pct=0.7,
                    sort_by="sharpe_ratio",
                    min_trades=2,
                )
                train_df = wf["train_results"]
                val_df = wf["validation_results"]
                best = wf["best_params"]
                overfit = wf["overfit_score"]

                if not val_df.empty:
                    top = val_df.iloc[0].to_dict()
                    top["strategy"] = strat_name
                    top["symbol"] = sym
                    top["overfit_score"] = overfit
                    all_results.append(top)
                    print(f"best Sharpe={top.get('sharpe_ratio', 'N/A')}, "
                          f"return={top.get('total_return_pct', 'N/A')}%, "
                          f"trades={top.get('total_trades', 0)}, "
                          f"overfit={overfit}%")
                else:
                    print("no valid results")
            except Exception as e:
                print(f"error: {e}")

    if not all_results:
        print("\nNo results found. Try relaxing constraints.")
        return

    print(f"\n[3/3] Ranking results...")
    results_df = pd.DataFrame(all_results)

    rank_cols = ["strategy", "symbol", "sharpe_ratio", "total_return_pct",
                 "max_drawdown_pct", "win_rate_pct", "total_trades",
                 "profit_factor", "calmar_ratio", "overfit_score"]
    display_cols = [c for c in rank_cols if c in results_df.columns]

    results_df = results_df.sort_values("sharpe_ratio", ascending=False, na_position="last")

    print("\n" + "=" * 80)
    print("TOP 10 STRATEGY-SYMBOL COMBINATIONS")
    print("=" * 80)
    top10 = results_df.head(10)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"\n  #{i}: {row.get('strategy', '?')} @ {row.get('symbol', '?')}")
        print(f"      Sharpe: {row.get('sharpe_ratio', 'N/A'):.4f}  |  "
              f"Return: {row.get('total_return_pct', 0):+.2f}%  |  "
              f"MaxDD: {row.get('max_drawdown_pct', 0):.2f}%")
        print(f"      Trades: {row.get('total_trades', 0)}  |  "
              f"WinRate: {row.get('win_rate_pct', 0):.1f}%  |  "
              f"PF: {row.get('profit_factor', 0):.2f}  |  "
              f"Overfit: {row.get('overfit_score', 'N/A')}%")

    optimizer.save_results(results_df, output_path)
    print(f"\nFull results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUTU-QUANT Parameter Scan")
    parser.add_argument("--output", default="results/scan_results.csv", help="Output CSV path")
    args = parser.parse_args()
    run_scan(args.output)
