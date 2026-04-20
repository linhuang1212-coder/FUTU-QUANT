"""Validate intraday strategies on 8 YEARS of REAL 5min data (2018-05 ~ 2026-04).

This is the gold standard: ~2000 trading days of real market microstructure.
Only strategies that pass here should go to production.

Strategies tested:
  - Rebalance 2pm (leveraged ETF rebalancing effect)
  - VWAP Trend Following
  - Opening Range Breakout
  - SMA200 Filtered Intraday (only trade on trend days)
"""

import sys, io, math
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data.downloader import load_5min, load_daily
from data.indicators import TechnicalIndicators


def precompute(df):
    out = df.copy()
    out = TechnicalIndicators.add_rsi(out, 14)
    out = TechnicalIndicators.add_ema(out, 8)
    out = TechnicalIndicators.add_ema(out, 20)
    out = TechnicalIndicators.add_atr(out, 14)
    tp = (out["high"] + out["low"] + out["close"]) / 3
    cum_tp_vol = (tp * out["volume"]).cumsum()
    cum_vol = out["volume"].cumsum().replace(0, np.nan)
    out["vwap"] = cum_tp_vol / cum_vol
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def split_days(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["time_key"]).dt.date
    days = []
    for _, g in df.groupby("date"):
        d = g.reset_index(drop=True)
        if len(d) >= 20:
            days.append(d)
    return days


def compute_metrics(equities, n_trades):
    eq = np.array(equities, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return {"sharpe": 0, "return_pct": 0, "max_dd": 0, "trades": 0, "win_rate": 0, "pf": 0, "cagr": 0}

    rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
    rf = 1.05 ** (1/252) - 1
    std = float(np.std(rets, ddof=1))
    sharpe = float(np.mean(rets - rf) / std * math.sqrt(252)) if std > 1e-12 else 0

    total_ret = (eq[-1] / eq[0] - 1) * 100
    years = len(eq) / 252
    cagr = ((eq[-1] / eq[0]) ** (1/years) - 1) * 100 if years > 0 and eq[-1] > 0 else 0

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd)) * 100

    return {
        "sharpe": round(sharpe, 3),
        "return_pct": round(total_ret, 1),
        "max_dd": round(max_dd, 1),
        "cagr": round(cagr, 1),
        "trades": n_trades,
    }


# ── Intraday Strategies ──────────────────────────────────────

def sim_rebalance_2pm(day_df, params):
    """Leveraged ETF end-of-day rebalancing effect."""
    move_t = params.get("move_threshold_pct", 1.5)
    entry_idx = params.get("entry_bar_idx", 48)
    stop_pct = params.get("stop_pct", 1.0)
    if len(day_df) <= entry_idx + 3:
        return None
    op = day_df["open"].iloc[0]
    p = day_df["close"].iloc[entry_idx]
    move = (p - op) / op * 100
    if move < move_t:
        return None
    ep = p * 1.0005
    sp = ep * (1 - stop_pct / 100)
    for i in range(entry_idx + 1, len(day_df)):
        if day_df["low"].iloc[i] <= sp:
            return (sp * 0.9995 - ep) / ep * 100
    return (day_df["close"].iloc[-1] * 0.9995 - ep) / ep * 100


def sim_vwap_trend(day_df, params):
    """VWAP trend following."""
    confirm = params.get("confirm_bars", 3)
    rsi_floor = params.get("rsi_floor", 50)
    min_bars = params.get("min_bars_before_entry", 6)
    stop_pct = params.get("stop_pct", 1.0)
    if len(day_df) < min_bars + confirm + 5 or "vwap" not in day_df.columns:
        return None
    close = day_df["close"].values
    vwap = day_df["vwap"].values
    rsi = day_df["rsi_14"].values if "rsi_14" in day_df.columns else np.full(len(day_df), 50)
    pos = 0
    ep = 0.0
    for i in range(min_bars, len(day_df)):
        if np.isnan(vwap[i]) or np.isnan(rsi[i]):
            continue
        if pos == 0:
            above = sum(1 for k in range(confirm) if i - k >= 0 and close[i - k] > vwap[i - k])
            if above >= confirm and rsi[i] > rsi_floor:
                ep = close[i] * 1.0005
                pos = 1
        else:
            if close[i] < vwap[i] or close[i] < ep * (1 - stop_pct / 100):
                return (close[i] * 0.9995 - ep) / ep * 100
    if pos > 0:
        return (day_df["close"].iloc[-1] * 0.9995 - ep) / ep * 100
    return None


def sim_orb(day_df, params):
    """Opening Range Breakout."""
    orb_bars = params.get("orb_bars", 6)
    vol_mult = params.get("vol_mult", 1.0)
    stop_pct = params.get("stop_pct", 1.5)
    if len(day_df) < orb_bars + 10:
        return None
    orb_high = day_df["high"].iloc[:orb_bars].max()
    orb_low = day_df["low"].iloc[:orb_bars].min()
    vol_avg = day_df["volume"].iloc[:orb_bars].mean()
    pos = 0
    ep = 0.0
    for i in range(orb_bars, len(day_df)):
        c = day_df["close"].iloc[i]
        v = day_df["volume"].iloc[i]
        if pos == 0:
            if c > orb_high and v > vol_mult * vol_avg:
                ep = c * 1.0005
                pos = 1
        else:
            if c < ep * (1 - stop_pct / 100):
                return (c * 0.9995 - ep) / ep * 100
    if pos > 0:
        return (day_df["close"].iloc[-1] * 0.9995 - ep) / ep * 100
    return None


def sim_mean_reversion_intraday(day_df, params):
    """Intraday mean reversion: buy on gap down that recovers above VWAP."""
    gap_pct = params.get("gap_threshold_pct", -1.5)
    stop_pct = params.get("stop_pct", 2.0)
    if len(day_df) < 20 or "vwap" not in day_df.columns:
        return None
    op = day_df["open"].iloc[0]
    prev_close = day_df["close"].iloc[0]
    gap = (op - prev_close) / prev_close * 100 if prev_close > 0 else 0
    if gap > gap_pct:
        return None
    pos = 0
    ep = 0.0
    for i in range(3, len(day_df)):
        c = day_df["close"].iloc[i]
        vwap = day_df["vwap"].iloc[i]
        if np.isnan(vwap):
            continue
        if pos == 0:
            if c > vwap and day_df["close"].iloc[i-1] <= day_df["vwap"].iloc[i-1]:
                ep = c * 1.0005
                pos = 1
        else:
            if c < ep * (1 - stop_pct / 100):
                return (c * 0.9995 - ep) / ep * 100
    if pos > 0:
        return (day_df["close"].iloc[-1] * 0.9995 - ep) / ep * 100
    return None


def sim_momentum_close(day_df, params):
    """Late-day momentum: if close > open by threshold at bar X, hold to close."""
    move_t = params.get("move_threshold_pct", 2.0)
    entry_idx = params.get("entry_bar_idx", 60)
    stop_pct = params.get("stop_pct", 0.8)
    if len(day_df) <= entry_idx + 2:
        return None
    op = day_df["open"].iloc[0]
    p = day_df["close"].iloc[entry_idx]
    move = (p - op) / op * 100
    if move < move_t:
        return None
    ep = p * 1.0005
    sp = ep * (1 - stop_pct / 100)
    for i in range(entry_idx + 1, len(day_df)):
        if day_df["low"].iloc[i] <= sp:
            return (sp * 0.9995 - ep) / ep * 100
    return (day_df["close"].iloc[-1] * 0.9995 - ep) / ep * 100


# ── Backtest Engine ──────────────────────────────────────────

def backtest_strategy(days, sim_fn, params, capital=10000.0):
    eq = capital
    equities = [capital]
    trades = []
    for d in days:
        pnl = sim_fn(d, params)
        if pnl is not None:
            eq += eq * 0.95 * pnl / 100
            trades.append(pnl)
        equities.append(eq)
    return equities, trades


# ── Main ──────────────────────────────────────────────────────

STRATEGIES = [
    ("Rebalance_2pm (1.5%/48/1.0%)", sim_rebalance_2pm,
     {"move_threshold_pct": 1.5, "entry_bar_idx": 48, "stop_pct": 1.0}),
    ("Rebalance_2pm (1.0%/48/1.0%)", sim_rebalance_2pm,
     {"move_threshold_pct": 1.0, "entry_bar_idx": 48, "stop_pct": 1.0}),
    ("Rebalance_2pm (2.0%/54/1.5%)", sim_rebalance_2pm,
     {"move_threshold_pct": 2.0, "entry_bar_idx": 54, "stop_pct": 1.5}),
    ("VWAP_Trend (3/50/6/1.0%)", sim_vwap_trend,
     {"confirm_bars": 3, "rsi_floor": 50, "min_bars_before_entry": 6, "stop_pct": 1.0}),
    ("VWAP_Trend (3/45/6/1.5%)", sim_vwap_trend,
     {"confirm_bars": 3, "rsi_floor": 45, "min_bars_before_entry": 6, "stop_pct": 1.5}),
    ("ORB (6bar/1.0x/1.5%)", sim_orb,
     {"orb_bars": 6, "vol_mult": 1.0, "stop_pct": 1.5}),
    ("ORB (6bar/1.2x/1.0%)", sim_orb,
     {"orb_bars": 6, "vol_mult": 1.2, "stop_pct": 1.0}),
    ("MeanRev_VWAP (-1.5%/2.0%)", sim_mean_reversion_intraday,
     {"gap_threshold_pct": -1.5, "stop_pct": 2.0}),
    ("MeanRev_VWAP (-2.0%/2.5%)", sim_mean_reversion_intraday,
     {"gap_threshold_pct": -2.0, "stop_pct": 2.5}),
    ("Momentum_Close (2%/60/0.8%)", sim_momentum_close,
     {"move_threshold_pct": 2.0, "entry_bar_idx": 60, "stop_pct": 0.8}),
    ("Momentum_Close (1.5%/54/1.0%)", sim_momentum_close,
     {"move_threshold_pct": 1.5, "entry_bar_idx": 54, "stop_pct": 1.0}),
]


def main():
    print("=" * 80)
    print("  8-YEAR REAL 5min DATA VALIDATION (2018-05 ~ 2026-04)")
    print("=" * 80)

    for sym in ["TQQQ", "SOXL"]:
        df = load_5min(sym)
        if df is None:
            print(f"\n{sym}: no 5min data")
            continue

        n_days_raw = len(set(pd.to_datetime(df["time_key"]).dt.date))
        print(f"\n{'='*70}")
        print(f"  {sym}: {len(df)} bars, {n_days_raw} trading days")
        print(f"{'='*70}")

        df = precompute(df)
        all_days = split_days(df)
        print(f"  {len(all_days)} complete trading days")

        # Full period
        print(f"\n  --- Full Period ({len(all_days)} days) ---")
        print(f"  {'Strategy':<35} {'Sharpe':>7} {'CAGR':>7} {'Return':>8} {'MaxDD':>7} {'Trades':>7}")
        print(f"  {'-'*72}")

        for name, fn, params in STRATEGIES:
            eq, trades = backtest_strategy(all_days, fn, params)
            m = compute_metrics(eq, len(trades))
            tag = " ***" if m["sharpe"] > 1.0 else (" **" if m["sharpe"] > 0.5 else "")
            print(f"  {name:<35} {m['sharpe']:>7.3f} {m['cagr']:>6.1f}% {m['return_pct']:>7.1f}% {m['max_dd']:>6.1f}% {len(trades):>7}{tag}")

        # Segmented: first 4yr vs last 4yr
        mid = len(all_days) // 2
        first_half = all_days[:mid]
        second_half = all_days[mid:]

        print(f"\n  --- First Half ({len(first_half)} days) vs Second Half ({len(second_half)} days) ---")
        print(f"  {'Strategy':<35} {'H1_Sh':>7} {'H2_Sh':>7} {'H1_Tr':>6} {'H2_Tr':>6} {'Consistent':>10}")
        print(f"  {'-'*75}")

        for name, fn, params in STRATEGIES:
            eq1, t1 = backtest_strategy(first_half, fn, params)
            eq2, t2 = backtest_strategy(second_half, fn, params)
            m1 = compute_metrics(eq1, len(t1))
            m2 = compute_metrics(eq2, len(t2))
            consistent = "YES" if m1["sharpe"] > 0 and m2["sharpe"] > 0 else "NO"
            print(f"  {name:<35} {m1['sharpe']:>7.3f} {m2['sharpe']:>7.3f} {len(t1):>6} {len(t2):>6} {consistent:>10}")

        # Walk-Forward: 3x rolling windows (60%/40% split)
        chunk_size = len(all_days) // 3
        print(f"\n  --- Rolling Walk-Forward (3 windows, ~{chunk_size} days each) ---")
        print(f"  {'Strategy':<35} {'W1_Sh':>7} {'W2_Sh':>7} {'W3_Sh':>7} {'Pass':>6}")
        print(f"  {'-'*65}")

        for name, fn, params in STRATEGIES:
            sharpes = []
            for w in range(3):
                w_days = all_days[w * chunk_size: min((w+1) * chunk_size, len(all_days))]
                split = int(len(w_days) * 0.6)
                test_days = w_days[split:]
                eq_t, tt = backtest_strategy(test_days, fn, params)
                m_t = compute_metrics(eq_t, len(tt))
                sharpes.append(m_t["sharpe"])
            n_pos = sum(1 for s in sharpes if s > 0)
            verdict = f"{n_pos}/3"
            print(f"  {name:<35} {sharpes[0]:>7.3f} {sharpes[1]:>7.3f} {sharpes[2]:>7.3f} {verdict:>6}")


if __name__ == "__main__":
    main()
