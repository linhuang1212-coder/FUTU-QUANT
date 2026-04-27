"""Backtest: 7-ETF momentum rotation vs old 2-ETF (TQQQ/SOXL) baseline.

Tests:
  1. Baseline: TQQQ buy-and-hold
  2. Old: 2-ETF momentum rotation (TQQQ vs SOXL)
  3. New: 7-ETF momentum rotation (top-3 candidates)

Segmented validation: 10yr, 5yr, 3yr + stress periods.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data_store/market_data")

POOL_OLD = ["TQQQ", "SOXL"]
POOL_NEW = ["TQQQ", "SOXL", "UPRO", "TNA", "TECL", "FAS", "LABU"]

TOP_N = 3
MOM_1M_WEIGHT = 0.5
MOM_3M_WEIGHT = 0.5
REBAL_DAYS = 21


def load_daily(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_daily.csv"
    df = pd.read_csv(path, parse_dates=["time_key"])
    df = df.sort_values("time_key").reset_index(drop=True)
    return df


def compute_momentum_scores(pool: list[str], date_idx: int,
                            all_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    scores = {}
    for sym in pool:
        df = all_data[sym]
        mask = df.index <= date_idx
        sub = df[mask]
        if len(sub) < 63:
            continue
        close = sub["close"].values
        mom_1m = close[-1] / close[-21] - 1 if len(close) >= 21 else 0
        mom_3m = close[-1] / close[-63] - 1 if len(close) >= 63 else 0
        scores[sym] = MOM_1M_WEIGHT * mom_1m + MOM_3M_WEIGHT * mom_3m
    return scores


def backtest_rotation(pool: list[str], all_data: dict[str, pd.DataFrame],
                      start_date: str, end_date: str,
                      top_n: int = TOP_N) -> dict:
    ref = all_data[pool[0]]
    mask = (ref["time_key"] >= start_date) & (ref["time_key"] <= end_date)
    indices = ref[mask].index.tolist()

    if len(indices) < 63:
        return {"sharpe": 0, "cagr": 0, "maxdd": 0, "trades": 0}

    portfolio_value = 10000.0
    values = [portfolio_value]
    current_symbol = None
    days_since_rebal = 0
    trade_count = 0

    for i, idx in enumerate(indices[63:], start=63):
        actual_idx = indices[i]

        # Monthly rebalance
        days_since_rebal += 1
        if days_since_rebal >= REBAL_DAYS or current_symbol is None:
            scores = compute_momentum_scores(pool, actual_idx, all_data)
            if not scores:
                values.append(portfolio_value)
                continue

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            candidates = [s for s, sc in ranked[:top_n] if sc > 0]

            new_sym = candidates[0] if candidates else None

            if new_sym != current_symbol:
                current_symbol = new_sym
                trade_count += 1
            days_since_rebal = 0

        # Daily return
        if current_symbol is not None:
            sym_df = all_data[current_symbol]
            if actual_idx > 0 and actual_idx < len(sym_df) and actual_idx - 1 >= 0:
                today_close = sym_df.loc[actual_idx, "close"]
                prev_close = sym_df.loc[actual_idx - 1, "close"]
                if prev_close > 0:
                    daily_ret = today_close / prev_close - 1
                    portfolio_value *= (1 + daily_ret)

        values.append(portfolio_value)

    values = np.array(values)
    if len(values) < 2:
        return {"sharpe": 0, "cagr": 0, "maxdd": 0, "trades": trade_count}

    returns = np.diff(values) / values[:-1]
    returns = returns[np.isfinite(returns)]

    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    years = len(returns) / 252
    cagr = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 and values[0] > 0 else 0
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / np.where(peak > 0, peak, 1)
    maxdd = dd.min()

    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr * 100, 1),
        "maxdd": round(maxdd * 100, 1),
        "trades": trade_count,
        "final_value": round(values[-1], 0),
    }


def backtest_buy_hold(symbol: str, all_data: dict[str, pd.DataFrame],
                      start_date: str, end_date: str) -> dict:
    df = all_data[symbol]
    mask = (df["time_key"] >= start_date) & (df["time_key"] <= end_date)
    sub = df[mask].reset_index(drop=True)

    if len(sub) < 20:
        return {"sharpe": 0, "cagr": 0, "maxdd": 0}

    prices = sub["close"].values
    returns = np.diff(prices) / prices[:-1]
    returns = returns[np.isfinite(returns)]

    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    years = len(returns) / 252
    cagr = (prices[-1] / prices[0]) ** (1 / years) - 1 if years > 0 and prices[0] > 0 else 0
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / np.where(peak > 0, peak, 1)
    maxdd = dd.min()

    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr * 100, 1),
        "maxdd": round(maxdd * 100, 1),
    }


def main():
    print("=" * 70)
    print("7-ETF Momentum Rotation Backtest")
    print("=" * 70)

    all_data = {}
    for sym in set(POOL_NEW + ["TQQQ"]):
        all_data[sym] = load_daily(sym)
        print(f"  Loaded {sym}: {len(all_data[sym])} bars "
              f"({all_data[sym]['time_key'].iloc[0].date()} to "
              f"{all_data[sym]['time_key'].iloc[-1].date()})")

    # Align all data to common date index
    ref_dates = set(all_data["TQQQ"]["time_key"].dt.date)
    for sym in all_data:
        all_data[sym] = all_data[sym][
            all_data[sym]["time_key"].dt.date.isin(ref_dates)
        ].reset_index(drop=True)

    segments = [
        ("10yr (full)", "2016-05-01", "2026-04-17"),
        ("5yr",         "2021-04-01", "2026-04-17"),
        ("3yr",         "2023-04-01", "2026-04-17"),
        ("COVID crash", "2020-01-01", "2020-06-30"),
        ("2022 bear",   "2022-01-01", "2022-12-31"),
        ("2024 rally",  "2024-01-01", "2024-12-31"),
    ]

    top_n_variants = [1, 2, 3]
    pool_variants = [
        ("7-ETF (all)", POOL_NEW),
        ("5-ETF (no LABU/FAS)", ["TQQQ", "SOXL", "UPRO", "TNA", "TECL"]),
        ("4-ETF (top performers)", ["TQQQ", "SOXL", "TECL", "UPRO"]),
    ]

    for seg_name, start, end in segments:
        print(f"\n{'─' * 70}")
        print(f"  {seg_name}  ({start} to {end})")
        print(f"{'─' * 70}")

        bh = backtest_buy_hold("TQQQ", all_data, start, end)
        old = backtest_rotation(POOL_OLD, all_data, start, end, top_n=1)

        print(f"  {'Strategy':<35} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Trades':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'TQQQ Buy&Hold':<35} {bh['sharpe']:>8.3f} {bh['cagr']:>7.1f}% {bh['maxdd']:>7.1f}%")
        print(f"  {'2-ETF Rotation (baseline)':<35} {old['sharpe']:>8.3f} {old['cagr']:>7.1f}% {old['maxdd']:>7.1f}% {old['trades']:>7}")

        for pool_label, pool in pool_variants:
            for tn in top_n_variants:
                label = f"{pool_label} top-{tn}"
                r = backtest_rotation(pool, all_data, start, end, top_n=tn)
                diff = r["sharpe"] - old["sharpe"]
                marker = " ***" if diff > 0.05 else ""
                print(f"  {label:<35} {r['sharpe']:>8.3f} {r['cagr']:>7.1f}% {r['maxdd']:>7.1f}% {r['trades']:>7}{marker}")

    # Per-ETF standalone performance for reference
    print(f"\n{'=' * 70}")
    print("Per-ETF standalone performance (10yr buy-and-hold)")
    print(f"{'=' * 70}")
    print(f"  {'ETF':<10} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for sym in POOL_NEW:
        r = backtest_buy_hold(sym, all_data, "2016-05-01", "2026-04-17")
        print(f"  {sym:<10} {r['sharpe']:>8.3f} {r['cagr']:>7.1f}% {r['maxdd']:>7.1f}%")


if __name__ == "__main__":
    main()
