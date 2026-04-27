"""Fast Walk-Forward optimization for UPRO and TECL.

Precomputes indicators once, then runs strategy on_bar over smaller windows
with reduced grid sizes for practical runtime (~5-10 min total).
"""

import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from strategy.breakout import BreakoutStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.multi_factor import MultiFactorStrategy
from data.indicators import TechnicalIndicators

DATA_DIR = Path("data_store/market_data")
TARGETS = ["UPRO", "TECL"]
INITIAL_CAPITAL = 3000.0
HARD_STOP = -0.08
WIN = 60
TRAIN_DAYS = 756   # ~3 years
TEST_DAYS = 252    # ~1 year


def load(sym):
    return pd.read_csv(
        DATA_DIR / f"{sym}_daily.csv", parse_dates=["time_key"]
    ).sort_values("time_key").reset_index(drop=True)


def add_indicators(df):
    o = df.copy()
    for p in (5, 8, 10, 14, 15, 20, 25):
        o = TechnicalIndicators.add_ma(o, p)
        o = TechnicalIndicators.add_ema(o, p)
    for p in (5, 7, 10, 14):
        o = TechnicalIndicators.add_rsi(o, p)
    for bp in (15, 20, 25):
        for bs in (1.5, 2.0, 2.5):
            o = TechnicalIndicators.add_bollinger(o, bp, bs)
    o = TechnicalIndicators.add_atr(o, 14)
    o = TechnicalIndicators.add_macd(o, 12, 26, 9)
    return o


def fast_backtest(df_ind, strat, start_idx, end_idx):
    """Backtest on pre-computed indicators DataFrame. Returns Sharpe."""
    closes = df_ind["close"].values
    n = end_idx - start_idx
    if n < WIN + 20:
        return 0.0

    cap = INITIAL_CAPITAL
    eq = [cap]
    pos = None

    for i in range(max(start_idx, start_idx + WIN), end_idx):
        p = closes[i]
        if pos and (p / pos[0] - 1) <= HARD_STOP:
            cap += pos[1] * (p - pos[0])
            pos = None

        try:
            sig = strat.on_bar("TEST", df_ind.iloc[i - WIN:i + 1])
        except Exception:
            sig = None

        if pos and sig and sig.direction.value == "SELL":
            cap += pos[1] * (p - pos[0])
            pos = None
        elif not pos and sig and sig.direction.value == "BUY":
            q = int(cap * 0.95 / p)
            if q > 0:
                pos = (p, q)

        eq.append(cap + pos[1] * (p - pos[0]) if pos else cap)

    v = np.array(eq)
    if len(v) < 20:
        return 0.0
    r = np.diff(v) / v[:-1]
    r = r[np.isfinite(r)]
    if np.std(r) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r) * np.sqrt(252))


PARAM_GRIDS = {
    "breakout": {
        "cls": BreakoutStrategy,
        "grid": {
            "lookback_period": [8, 10, 15],
            "volume_ratio_threshold": [1.0, 1.2, 1.5],
            "atr_breakout_multiplier": [1.0, 1.5, 2.0],
        },
    },
    "mean_reversion": {
        "cls": MeanReversionStrategy,
        "grid": {
            "bb_period": [15, 20],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_period": [10, 14],
            "rsi_oversold": [20, 25, 30],
            "rsi_overbought": [70, 75],
        },
    },
    "multi_factor": {
        "cls": MultiFactorStrategy,
        "grid": {
            "fast_ma_period": [5, 8, 10],
            "slow_ma_period": [15, 20],
            "rsi_period": [10, 14],
            "ema_period": [15, 20],
            "buy_threshold": [3, 4],
            "sell_threshold": [3],
        },
    },
}


def walk_forward(sym, df_ind, strat_name, cls, param_combos):
    n = len(df_ind)
    window_size = TRAIN_DAYS + TEST_DAYS
    step = TEST_DAYS
    n_windows = min(5, (n - window_size) // step + 1)

    if n_windows < 3:
        print(f"  WARNING: Only {n_windows} windows for {sym}/{strat_name}")
        return None, 0, 0

    best_params = None
    best_avg_oos = -999
    best_consistency = 0

    total = len(param_combos)
    for pi, params in enumerate(param_combos):
        if pi % 20 == 0:
            print(f"    {strat_name}: {pi}/{total}...", flush=True)

        oos_sharpes = []
        for wi in range(n_windows):
            train_start = n - window_size - (n_windows - 1 - wi) * step
            train_end = train_start + TRAIN_DAYS
            test_end = train_end + TEST_DAYS
            if train_start < WIN or test_end > n:
                continue

            strat = cls(params=dict(params))
            is_sharpe = fast_backtest(df_ind, strat, train_start, train_end)
            if is_sharpe < 0.2:
                oos_sharpes.append(0)
                continue
            oos_sharpe = fast_backtest(df_ind, strat, train_end, test_end)
            oos_sharpes.append(oos_sharpe)

        if len(oos_sharpes) < 3:
            continue
        avg_oos = np.mean(oos_sharpes)
        consistency = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)

        if avg_oos > best_avg_oos and consistency >= 0.6:
            best_avg_oos = avg_oos
            best_params = dict(params)
            best_consistency = consistency

    return best_params, best_avg_oos, best_consistency


def main():
    for sym in TARGETS:
        print(f"\n{'='*70}")
        print(f"  Walk-Forward Optimization: {sym}")
        print(f"{'='*70}")
        df_raw = load(sym)
        df_ind = add_indicators(df_raw)
        print(f"  Data: {len(df_raw)} bars ({df_raw['time_key'].iloc[0].date()} ~ {df_raw['time_key'].iloc[-1].date()})")
        print(f"  Windows: 5 x ({TRAIN_DAYS}d train + {TEST_DAYS}d test)")

        for sname, scfg in PARAM_GRIDS.items():
            cls = scfg["cls"]
            grid = scfg["grid"]
            keys = list(grid.keys())
            values = list(grid.values())
            combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
            print(f"\n  Strategy: {sname} ({len(combos)} param combos)")

            best_params, avg_oos, consistency = walk_forward(
                sym, df_ind, sname, cls, combos
            )
            if best_params:
                print(f"  >>> BEST: avg OOS Sharpe={avg_oos:.3f}, consistency={consistency:.0%}")
                print(f"  >>> Params: {best_params}")
            else:
                print(f"  >>> No params passed consistency filter (>=60%)")

    print(f"\n{'='*70}")
    print("  Done! Update config/live.yaml with the best params above.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
