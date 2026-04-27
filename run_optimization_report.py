"""Compare original baseline vs optimized system performance.

Runs the optimized backtest and displays a side-by-side comparison
against the pre-optimization baseline numbers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from run_full_system_backtest import (
    load, add_indicators, precompute_all_signals,
    bt_rotation, calc, POOL, INITIAL_CAPITAL, SIG_FILTER,
)

SEGMENTS = [
    ("10yr", "2016-05-01", "2026-04-17"),
    ("5yr",  "2021-04-01", "2026-04-17"),
    ("3yr",  "2023-04-01", "2026-04-17"),
    ("1yr",  "2025-04-01", "2026-04-17"),
]

BASELINE = {
    "10yr": {"sharpe": 0.881, "cagr": 25.1, "maxdd": -39.0, "trades": 1207, "final": 27690},
    "5yr":  {"sharpe": 0.838, "cagr": 23.3, "maxdd": -38.6, "trades": 503,  "final": 8588},
    "3yr":  {"sharpe": 1.136, "cagr": 37.2, "maxdd": -34.9, "trades": 345,  "final": 7820},
    "1yr":  {"sharpe": 2.219, "cagr": 96.6, "maxdd": -15.5, "trades": 113,  "final": 6076},
}


def main():
    print("=" * 90)
    print("  FUTU-QUANT Optimization Report: Baseline vs Optimized")
    print("=" * 90)

    print("\nLoading data & precomputing signals...")
    all_raw, all_ind = {}, {}
    for sym in POOL:
        all_raw[sym] = load(sym)
        all_ind[sym] = add_indicators(all_raw[sym])

    all_buy, all_sell = {}, {}
    for sym in POOL:
        b, s = precompute_all_signals(sym, all_ind[sym])
        all_buy[sym] = b
        all_sell[sym] = s

    all_buy, all_sell = SIG_FILTER.filter_signals_vectorized(
        all_buy, all_sell, all_ind, POOL
    )

    print("\n" + "=" * 90)
    hdr = f"  {'Segment':<8} {'Metric':<8} {'Baseline':>10} {'Optimized':>10} {'Delta':>10} {'Status':>8}"
    print(hdr)
    print("-" * 90)

    all_pass = True
    for seg_name, start, end in SEGMENTS:
        r = bt_rotation(all_raw, all_buy, all_sell, start, end, sma200=True)
        b = BASELINE[seg_name]

        for metric, label in [("sharpe", "Sharpe"), ("cagr", "CAGR%"), ("maxdd", "MaxDD%")]:
            bv = b[metric]
            ov = r[metric]
            delta = ov - bv

            if metric == "maxdd":
                ok = delta >= 0  # less negative = better
            else:
                ok = delta >= -0.05 * abs(bv)  # allow 5% regression

            status = "OK" if ok else "WARN"
            if not ok:
                all_pass = False

            sign = "+" if delta > 0 else ""
            print(f"  {seg_name:<8} {label:<8} {bv:>10.1f} {ov:>10.1f} {sign}{delta:>9.1f} {status:>8}")

        bfinal = b["final"]
        ofinal = r["final"]
        delta_f = ofinal - bfinal
        sign_f = "+" if delta_f > 0 else ""
        print(f"  {seg_name:<8} {'$Final':<8} {bfinal:>10,.0f} {ofinal:>10,.0f} {sign_f}{delta_f:>9,.0f}")

        btrades = b["trades"]
        otrades = r["trades"]
        delta_t = otrades - btrades
        sign_t = "+" if delta_t > 0 else ""
        print(f"  {seg_name:<8} {'Trades':<8} {btrades:>10} {otrades:>10} {sign_t}{delta_t:>9}")

        print(f"  {seg_name:<8} {'WinRate':<8} {'':>10} {r['win_rate']:>9.1f}%")
        print("-" * 90)

    print()
    print("  Optimizations applied:")
    print("    1. Dynamic position sizing (signal strength + momentum)")
    print("    2. Signal quality filter (BUY-only, strength>=60, volume confirm)")
    print("    3. Risk-adjusted momentum rotation + hysteresis")
    print("    4. Tiered trailing stop (+10%/6%, +25%/4%)")
    print("    5. Cash yield (4.5% annualized during flat periods)")
    print()
    if all_pass:
        print("  >>> ALL SEGMENTS PASS: Safe to deploy <<<")
    else:
        print("  >>> SOME WARNINGS: Review before deploying <<<")
    print("=" * 90)


if __name__ == "__main__":
    main()
