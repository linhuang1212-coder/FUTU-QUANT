"""Validate new macro strategies using REAL daily data only.

Strategies:
  1. QQQ SMA200 Trend Filter  — hold TQQQ/SOXL only when QQQ > SMA200
  2. Dual Momentum Rotation   — monthly rotation between TQQQ/SOXL/cash
  3. VIX Adaptive Sizing      — continuous position sizing based on VIX level

Validation: 10yr / 5yr / 3yr segmented + stress periods + Buy & Hold comparison.
"""

import sys, io, math
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data.downloader import load_daily

SEGMENTS = {
    "10yr_full":  ("2016-01-01", "2026-04-30"),
    "5yr_recent": ("2021-01-01", "2026-04-30"),
    "3yr_recent": ("2023-01-01", "2026-04-30"),
}

STRESS_PERIODS = {
    "COVID_crash":    ("2020-01-01", "2020-06-30"),
    "rate_hike_bear": ("2022-01-01", "2022-12-31"),
    "AI_bull":        ("2023-01-01", "2024-12-31"),
}


def load_all_data():
    """Load daily data for all needed symbols."""
    data = {}
    for sym in ["QQQ", "SPY", "TQQQ", "SOXL"]:
        df = load_daily(sym)
        if df is not None:
            df["time_key"] = pd.to_datetime(df["time_key"])
            df = df.sort_values("time_key").reset_index(drop=True)
            data[sym] = df
            print(f"  {sym}: {len(df)} days ({df['time_key'].iloc[0].date()} ~ {df['time_key'].iloc[-1].date()})")
        else:
            print(f"  {sym}: NO DATA")
    return data


def slice_data(df, start, end):
    mask = (df["time_key"] >= start) & (df["time_key"] <= end)
    return df[mask].reset_index(drop=True)


def compute_metrics(equities, years):
    """Compute Sharpe, CAGR, Max DD from equity curve."""
    eq = np.array(equities, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "final": eq[-1] if len(eq) > 0 else 0}

    rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
    rf_daily = 1.05 ** (1/252) - 1
    excess = rets - rf_daily
    std = float(np.std(rets, ddof=1))
    sharpe = float(np.mean(excess) / std * math.sqrt(252)) if std > 1e-12 else 0

    total_ret = eq[-1] / eq[0]
    cagr = (total_ret ** (1/years) - 1) * 100 if years > 0 and total_ret > 0 else 0

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd)) * 100

    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "final": round(eq[-1], 2),
    }


# ── Strategy 1: QQQ SMA200 Trend Filter ──────────────────────

def backtest_sma200_filter(qqq_df, etf_df, sma_period=200, capital=10000.0):
    """Hold ETF only when QQQ > SMA(200). Otherwise cash."""
    qqq = qqq_df.copy()
    etf = etf_df.copy()

    qqq[f"sma{sma_period}"] = qqq["close"].rolling(sma_period).mean()
    merged = pd.merge(
        etf[["time_key", "open", "close"]],
        qqq[["time_key", f"sma{sma_period}", "close"]].rename(
            columns={"close": "qqq_close"}),
        on="time_key", how="inner"
    )
    merged = merged.dropna(subset=[f"sma{sma_period}"]).reset_index(drop=True)

    eq = capital
    equities = [eq]
    in_market = False
    trades = 0

    for i in range(1, len(merged)):
        prev_trend = merged["qqq_close"].iloc[i-1] > merged[f"sma{sma_period}"].iloc[i-1]
        curr_ret = merged["close"].iloc[i] / merged["close"].iloc[i-1] - 1

        if in_market:
            eq *= (1 + curr_ret)
            if not prev_trend:
                in_market = False
                trades += 1
        else:
            if prev_trend:
                in_market = True
                trades += 1

        equities.append(eq)

    n_days = len(merged)
    years = n_days / 252
    return equities, trades, years


def backtest_buy_hold(etf_df, capital=10000.0):
    """Simple buy & hold benchmark."""
    closes = etf_df["close"].values
    eq = capital
    equities = [eq]
    for i in range(1, len(closes)):
        eq *= closes[i] / closes[i-1]
        equities.append(eq)
    years = len(closes) / 252
    return equities, years


# ── Strategy 2: Dual Momentum Rotation ────────────────────────

def backtest_dual_momentum(tqqq_df, soxl_df, capital=10000.0,
                           lookback_1m=21, lookback_3m=63,
                           rebal_freq=21):
    """Monthly rotation: rank TQQQ vs SOXL by weighted momentum, hold winner.
    If both negative -> cash."""
    merged = pd.merge(
        tqqq_df[["time_key", "close"]].rename(columns={"close": "tqqq"}),
        soxl_df[["time_key", "close"]].rename(columns={"close": "soxl"}),
        on="time_key", how="inner"
    )
    merged = merged.sort_values("time_key").reset_index(drop=True)

    eq = capital
    equities = [eq]
    holding = "cash"  # "tqqq", "soxl", "cash"
    trades = 0

    for i in range(1, len(merged)):
        if holding == "tqqq":
            eq *= merged["tqqq"].iloc[i] / merged["tqqq"].iloc[i-1]
        elif holding == "soxl":
            eq *= merged["soxl"].iloc[i] / merged["soxl"].iloc[i-1]

        equities.append(eq)

        if i % rebal_freq == 0 and i >= lookback_3m:
            mom_tqqq_1m = merged["tqqq"].iloc[i] / merged["tqqq"].iloc[i - lookback_1m] - 1
            mom_tqqq_3m = merged["tqqq"].iloc[i] / merged["tqqq"].iloc[i - lookback_3m] - 1
            mom_soxl_1m = merged["soxl"].iloc[i] / merged["soxl"].iloc[i - lookback_1m] - 1
            mom_soxl_3m = merged["soxl"].iloc[i] / merged["soxl"].iloc[i - lookback_3m] - 1

            score_tqqq = 0.5 * mom_tqqq_1m + 0.5 * mom_tqqq_3m
            score_soxl = 0.5 * mom_soxl_1m + 0.5 * mom_soxl_3m

            if score_tqqq <= 0 and score_soxl <= 0:
                new_hold = "cash"
            elif score_tqqq >= score_soxl:
                new_hold = "tqqq"
            else:
                new_hold = "soxl"

            if new_hold != holding:
                trades += 1
                holding = new_hold

    years = len(merged) / 252
    return equities, trades, years


# ── Strategy 3: VIX Adaptive Sizing ───────────────────────────

def backtest_vix_adaptive(qqq_df, etf_df, capital=10000.0, sma_period=200):
    """SMA200 filter + continuous VIX-based position sizing.
    Since we don't have historical VIX in CSV, we proxy with
    realized volatility (20-day rolling) mapped to VIX-like levels.
    VIX proxy = 20d realized vol * sqrt(252) * 100
    """
    qqq = qqq_df.copy()
    etf = etf_df.copy()

    qqq[f"sma{sma_period}"] = qqq["close"].rolling(sma_period).mean()
    qqq["ret"] = qqq["close"].pct_change()
    qqq["rvol20"] = qqq["ret"].rolling(20).std() * np.sqrt(252) * 100

    merged = pd.merge(
        etf[["time_key", "close"]],
        qqq[["time_key", f"sma{sma_period}", "close", "rvol20"]].rename(
            columns={"close": "qqq_close"}),
        on="time_key", how="inner"
    )
    merged = merged.dropna(subset=[f"sma{sma_period}", "rvol20"]).reset_index(drop=True)

    eq = capital
    equities = [eq]
    trades = 0
    prev_alloc = 0.0

    for i in range(1, len(merged)):
        trend_ok = merged["qqq_close"].iloc[i-1] > merged[f"sma{sma_period}"].iloc[i-1]
        vix_proxy = merged["rvol20"].iloc[i-1]

        if not trend_ok:
            alloc = 0.0
        elif vix_proxy < 15:
            alloc = 0.95
        elif vix_proxy < 20:
            alloc = 0.75
        elif vix_proxy < 28:
            alloc = 0.50
        else:
            alloc = 0.0

        if alloc != prev_alloc:
            trades += 1
            prev_alloc = alloc

        daily_ret = merged["close"].iloc[i] / merged["close"].iloc[i-1] - 1
        eq *= (1 + alloc * daily_ret)
        equities.append(eq)

    years = len(merged) / 252
    return equities, trades, years


# ── Rolling Walk-Forward ──────────────────────────────────────

def rolling_wf_sma200(qqq_full, etf_full, train_years=3, test_years=1):
    """Rolling WF for SMA200 filter: we test different SMA periods."""
    periods_to_test = [150, 180, 200, 220, 250]
    start = qqq_full["time_key"].min()
    end = qqq_full["time_key"].max()

    window_start = start + pd.DateOffset(days=30)
    results = []

    while True:
        train_end = window_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end:
            break

        best_period = 200
        best_sharpe = -999

        for p in periods_to_test:
            qqq_train = slice_data(qqq_full, window_start, train_end)
            etf_train = slice_data(etf_full, window_start, train_end)
            if len(qqq_train) < p + 50:
                continue
            eq, _, yrs = backtest_sma200_filter(qqq_train, etf_train, sma_period=p)
            m = compute_metrics(eq, yrs)
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_period = p

        qqq_test = slice_data(qqq_full, train_end, test_end)
        etf_test = slice_data(etf_full, train_end, test_end)
        if len(qqq_test) > 50:
            eq, _, yrs = backtest_sma200_filter(qqq_test, etf_test, sma_period=best_period)
            m = compute_metrics(eq, yrs)
            results.append({
                "window": f"{train_end.date()}~{test_end.date()}",
                "best_period": best_period,
                "oos_sharpe": m["sharpe"],
                "oos_cagr": m["cagr"],
                "oos_maxdd": m["max_dd"],
            })

        window_start += pd.DateOffset(years=1)

    return results


# ── Main ──────────────────────────────────────────────────────

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print("=" * 70)
    print("  NEW DIRECTION VALIDATION — Real Daily Data Only")
    print("=" * 70)

    data = load_all_data()
    if "QQQ" not in data or "TQQQ" not in data:
        print("ERROR: Need at least QQQ and TQQQ daily data")
        return

    for etf_name in ["TQQQ", "SOXL"]:
        if etf_name not in data:
            continue

        # ── Strategy 1: SMA200 Filter ──
        print_header(f"Strategy 1: QQQ SMA200 Filter -> {etf_name}")
        print(f"  {'Segment':<20} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Trades':>8} {'BH_Sharpe':>10} {'BH_CAGR':>9}")
        print(f"  {'-'*73}")

        for seg_name, (s, e) in {**SEGMENTS, **STRESS_PERIODS}.items():
            qqq_seg = slice_data(data["QQQ"], s, e)
            etf_seg = slice_data(data[etf_name], s, e)
            if len(qqq_seg) < 250 or len(etf_seg) < 250:
                if len(qqq_seg) < 50 or len(etf_seg) < 50:
                    print(f"  {seg_name:<20} SKIP (insufficient data)")
                    continue

            eq, trades, yrs = backtest_sma200_filter(qqq_seg, etf_seg, 200)
            m = compute_metrics(eq, yrs)

            bh_eq, bh_yrs = backtest_buy_hold(etf_seg)
            bh = compute_metrics(bh_eq, bh_yrs)

            verdict = "PASS" if m["sharpe"] > 1.0 else ("ok" if m["sharpe"] > 0.5 else "FAIL")
            print(f"  {seg_name:<20} {m['sharpe']:>8.3f} {m['cagr']:>7.1f}% {m['max_dd']:>7.1f}% {trades:>8} {bh['sharpe']:>10.3f} {bh['cagr']:>8.1f}%  {verdict}")

        # Rolling WF
        print(f"\n  Rolling Walk-Forward (3yr train + 1yr test):")
        wf_results = rolling_wf_sma200(data["QQQ"], data[etf_name])
        if wf_results:
            n_pass = sum(1 for r in wf_results if r["oos_sharpe"] > 0.5)
            consistency = n_pass / len(wf_results)
            print(f"  {'Window':<25} {'Period':>7} {'OOS_Sharpe':>11} {'OOS_CAGR':>10} {'OOS_MaxDD':>10}")
            print(f"  {'-'*65}")
            for r in wf_results:
                v = "PASS" if r["oos_sharpe"] > 0.5 else "FAIL"
                print(f"  {r['window']:<25} {r['best_period']:>7} {r['oos_sharpe']:>11.3f} {r['oos_cagr']:>9.1f}% {r['oos_maxdd']:>9.1f}%  {v}")
            print(f"  Consistency: {n_pass}/{len(wf_results)} = {consistency:.0%}")
        else:
            print(f"  No walk-forward windows available")

    # ── Strategy 2: Dual Momentum ──
    if "TQQQ" in data and "SOXL" in data:
        print_header("Strategy 2: Dual Momentum Rotation (TQQQ/SOXL/Cash)")
        print(f"  {'Segment':<20} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Trades':>8}")
        print(f"  {'-'*55}")

        for seg_name, (s, e) in {**SEGMENTS, **STRESS_PERIODS}.items():
            tqqq_seg = slice_data(data["TQQQ"], s, e)
            soxl_seg = slice_data(data["SOXL"], s, e)
            if len(tqqq_seg) < 100 or len(soxl_seg) < 100:
                print(f"  {seg_name:<20} SKIP (insufficient data)")
                continue

            eq, trades, yrs = backtest_dual_momentum(tqqq_seg, soxl_seg)
            m = compute_metrics(eq, yrs)
            verdict = "PASS" if m["sharpe"] > 1.0 else ("ok" if m["sharpe"] > 0.5 else "FAIL")
            print(f"  {seg_name:<20} {m['sharpe']:>8.3f} {m['cagr']:>7.1f}% {m['max_dd']:>7.1f}% {trades:>8}  {verdict}")

    # ── Strategy 3: VIX Adaptive ──
    for etf_name in ["TQQQ", "SOXL"]:
        if etf_name not in data:
            continue

        print_header(f"Strategy 3: SMA200 + VIX Adaptive Sizing -> {etf_name}")
        print(f"  {'Segment':<20} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Trades':>8} {'SMA_Only':>10}")
        print(f"  {'-'*65}")

        for seg_name, (s, e) in {**SEGMENTS, **STRESS_PERIODS}.items():
            qqq_seg = slice_data(data["QQQ"], s, e)
            etf_seg = slice_data(data[etf_name], s, e)
            if len(qqq_seg) < 250 or len(etf_seg) < 250:
                if len(qqq_seg) < 50 or len(etf_seg) < 50:
                    print(f"  {seg_name:<20} SKIP")
                    continue

            eq_vix, trades_vix, yrs_vix = backtest_vix_adaptive(qqq_seg, etf_seg)
            m_vix = compute_metrics(eq_vix, yrs_vix)

            eq_sma, _, yrs_sma = backtest_sma200_filter(qqq_seg, etf_seg, 200)
            m_sma = compute_metrics(eq_sma, yrs_sma)

            verdict = "PASS" if m_vix["sharpe"] > 1.0 else ("ok" if m_vix["sharpe"] > 0.5 else "FAIL")
            print(f"  {seg_name:<20} {m_vix['sharpe']:>8.3f} {m_vix['cagr']:>7.1f}% {m_vix['max_dd']:>7.1f}% {trades_vix:>8} {m_sma['sharpe']:>10.3f}  {verdict}")

    # ── Summary ──
    print_header("SUMMARY")
    print("  Strategy 1 (SMA200 Filter):  QQQ > SMA200 hold ETF, otherwise cash")
    print("  Strategy 2 (Dual Momentum):  Monthly rank TQQQ/SOXL by 1m+3m momentum")
    print("  Strategy 3 (VIX Adaptive):   SMA200 + continuous position sizing by vol proxy")
    print()
    print("  Pass criteria: Sharpe > 1.0 across all time segments")
    print("  Strategies that pass will be deployed to run_live.py")


if __name__ == "__main__":
    main()
