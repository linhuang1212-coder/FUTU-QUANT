"""Full system vectorized backtest.

Precomputes all signals for all strategies/symbols in one pass,
then simulates the trading logic day-by-day using signal arrays.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from strategy.momentum import MomentumStrategy
from strategy.breakout import BreakoutStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.rsi_reversal import RsiReversalStrategy
from strategy.multi_factor import MultiFactorStrategy
from data.indicators import TechnicalIndicators
from strategy.signal_filter import SignalFilter

DATA_DIR = Path("data_store/market_data")
POOL = ["TQQQ", "SOXL", "UPRO", "TECL"]
HARD_STOP_PCT = -0.08
INITIAL_CAPITAL = 3000.0
WIN = 60

# Signal filter (Task 2) — disabled for clean comparison
SIG_FILTER = SignalFilter({"enabled": False})

# Trailing stop params (Task 5) — wide for leveraged ETFs
TS_ENABLED = False
TS_ACTIVATE = 0.15       # Only activate after +15% gain
TS_TRAIL = 0.08          # Trail 8% from peak (leveraged ETFs swing 5%+/day)
TS_TIER2_ACTIVATE = 0.30 # Tighten after +30% gain
TS_TIER2_TRAIL = 0.05    # Trail 5% from peak at tier 2

# Cash yield (Task 6)
DAILY_YIELD = (1 + 0.045) ** (1 / 252) - 1

STRAT_CFG = {
    "TQQQ": [
        ("momentum", MomentumStrategy, 1.22, {
            "fast_ma_period": 8, "slow_ma_period": 15, "rsi_period": 10,
            "rsi_oversold": 30, "rsi_overbought": 70, "volume_ratio_threshold": 1.0,
            "cross_lookback": 3, "rsi_momentum_enabled": True, "ema_trend_enabled": True}),
        ("breakout", BreakoutStrategy, 1.21, {
            "lookback_period": 10, "volume_ratio_threshold": 1.2, "atr_breakout_multiplier": 1.5}),
        ("mean_reversion", MeanReversionStrategy, 0.72, {
            "bb_period": 15, "bb_std": 2.0, "rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 75}),
        ("multi_factor", MultiFactorStrategy, 0.85, {
            "fast_ma_period": 10, "slow_ma_period": 15, "rsi_period": 10,
            "ema_period": 15, "buy_threshold": 3, "sell_threshold": 3}),
    ],
    "SOXL": [
        ("breakout", BreakoutStrategy, 1.20, {
            "lookback_period": 10, "volume_ratio_threshold": 1.2, "atr_breakout_multiplier": 1.5}),
        ("mean_reversion", MeanReversionStrategy, 1.06, {
            "bb_period": 20, "bb_std": 2.0, "rsi_period": 10, "rsi_oversold": 25, "rsi_overbought": 70}),
        ("rsi_reversal", RsiReversalStrategy, 0.98, {
            "rsi_period": 5, "rsi_buy_threshold": 25, "rsi_sell_threshold": 75}),
        ("multi_factor", MultiFactorStrategy, 0.63, {
            "fast_ma_period": 8, "slow_ma_period": 20, "rsi_period": 10,
            "ema_period": 20, "buy_threshold": 3, "sell_threshold": 3}),
    ],
    "UPRO": [
        ("breakout", BreakoutStrategy, 1.0, {
            "lookback_period": 10, "volume_ratio_threshold": 1.2, "atr_breakout_multiplier": 1.5}),
        ("mean_reversion", MeanReversionStrategy, 1.0, {
            "bb_period": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 75}),
        ("multi_factor", MultiFactorStrategy, 1.0, {
            "fast_ma_period": 10, "slow_ma_period": 20, "rsi_period": 10,
            "ema_period": 15, "buy_threshold": 3, "sell_threshold": 3}),
    ],
    "TECL": [
        ("breakout", BreakoutStrategy, 1.0, {
            "lookback_period": 10, "volume_ratio_threshold": 1.2, "atr_breakout_multiplier": 1.5}),
        ("mean_reversion", MeanReversionStrategy, 1.0, {
            "bb_period": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 75}),
        ("multi_factor", MultiFactorStrategy, 1.0, {
            "fast_ma_period": 10, "slow_ma_period": 20, "rsi_period": 10,
            "ema_period": 15, "buy_threshold": 3, "sell_threshold": 3}),
    ],
}


def load(sym): 
    return pd.read_csv(DATA_DIR / f"{sym}_daily.csv", parse_dates=["time_key"]).sort_values("time_key").reset_index(drop=True)


def add_indicators(df):
    o = df.copy()
    for p in (5, 8, 10, 14, 15, 20):
        o = TechnicalIndicators.add_ma(o, p)
        o = TechnicalIndicators.add_ema(o, p)
    for p in (5, 7, 10, 14):
        o = TechnicalIndicators.add_rsi(o, p)
    for bp, bs in ((15, 2.0), (20, 2.0)):
        o = TechnicalIndicators.add_bollinger(o, bp, bs)
    o = TechnicalIndicators.add_atr(o, 14)
    o = TechnicalIndicators.add_macd(o, 12, 26, 9)
    return o


def precompute_all_signals(sym, df_ind):
    """Run all strategies on every bar once, store results as arrays."""
    n = len(df_ind)
    strats = [(sn, cls(params=p), w) for sn, cls, w, p in STRAT_CFG[sym]]

    # For each day: best_buy_score, best_sell_score
    buy_scores = np.zeros(n)
    sell_scores = np.zeros(n)

    print(f"    Computing signals for {sym} ({n} bars, {len(strats)} strategies)...", flush=True)

    for i in range(WIN, n):
        window = df_ind.iloc[i - WIN:i + 1]
        for sn, st, w in strats:
            try:
                sig = st.on_bar(f"US.{sym}", window)
            except Exception:
                continue
            if sig is None:
                continue
            sc = w * sig.strength
            if sig.direction.value == "BUY" and sc > buy_scores[i]:
                buy_scores[i] = sc
            elif sig.direction.value == "SELL" and sc > sell_scores[i]:
                sell_scores[i] = sc

    return buy_scores, sell_scores


def calc(eq, trades):
    v = np.array(eq)
    if len(v) < 2:
        return dict(sharpe=0, cagr=0, maxdd=0, trades=0, final=0, win_rate=0)
    r = np.diff(v) / v[:-1]
    r = r[np.isfinite(r)]
    sh = (np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 0 else 0
    y = max(len(r) / 252, 0.01)
    cg = (v[-1] / v[0]) ** (1 / y) - 1
    pk = np.maximum.accumulate(v)
    dd = (v - pk) / np.where(pk > 0, pk, 1)
    return dict(sharpe=round(sh, 3), cagr=round(cg * 100, 1), maxdd=round(dd.min() * 100, 1),
                trades=trades, final=round(v[-1], 0), win_rate=round(np.sum(r > 0) / max(len(r), 1) * 100, 1))


def bt_per_symbol(sym, df_raw, buy_sc, sell_sc, start, end):
    m = (df_raw["time_key"] >= start) & (df_raw["time_key"] <= end)
    idx = df_raw[m].index.tolist()
    if len(idx) < WIN:
        return calc([INITIAL_CAPITAL], 0)

    closes = df_raw["close"].values
    cap = INITIAL_CAPITAL
    eq = [cap]
    pos = None
    tr = 0
    for i in idx:
        p = closes[i]
        if pos and (p / pos[0] - 1) <= HARD_STOP_PCT:
            cap += pos[1] * (p - pos[0])
            pos = None
            tr += 1
        if pos and sell_sc[i] > 0:
            cap += pos[1] * (p - pos[0])
            pos = None
            tr += 1
        elif not pos and buy_sc[i] > 0:
            q = int(cap * 0.95 / p)
            if q > 0:
                pos = (p, q)
                tr += 1
        eq.append(cap + pos[1] * (p - pos[0]) if pos else cap)
    return calc(eq, tr)


def bt_rotation(all_raw, all_buy, all_sell, start, end, sma200=True):
    ref = all_raw["TQQQ"]
    m = (ref["time_key"] >= start) & (ref["time_key"] <= end)
    date_idx = ref[m].index.tolist()
    if len(date_idx) < 80:
        return calc([INITIAL_CAPITAL], 0)

    qqq = load("QQQ").set_index("time_key")

    cap = INITIAL_CAPITAL
    eq = [cap]
    hold = None  # (symbol, entry_price, qty)
    tr = 0
    highest_price = 0.0  # trailing stop tracking

    for di, idx in enumerate(date_idx):
        if di < 63:
            eq.append(cap)
            continue

        day = ref.at[idx, "time_key"]
        closes = {s: all_raw[s].at[idx, "close"] if idx < len(all_raw[s]) else None for s in POOL}

        # Hard stop
        if hold and closes.get(hold[0]):
            if (closes[hold[0]] / hold[1] - 1) <= HARD_STOP_PCT:
                cap += hold[2] * (closes[hold[0]] - hold[1])
                hold = None
                highest_price = 0
                tr += 1

        # Trailing stop (Task 5)
        if TS_ENABLED and hold and closes.get(hold[0]):
            cur_p = closes[hold[0]]
            highest_price = max(highest_price, cur_p)
            pnl_from_entry = (highest_price / hold[1]) - 1.0
            ts_stop = 0
            if pnl_from_entry >= TS_TIER2_ACTIVATE:
                ts_stop = highest_price * (1 - TS_TIER2_TRAIL)
            elif pnl_from_entry >= TS_ACTIVATE:
                ts_stop = highest_price * (1 - TS_TRAIL)
            if ts_stop > 0 and cur_p <= ts_stop:
                cap += hold[2] * (cur_p - hold[1])
                hold = None
                highest_price = 0
                tr += 1

        # SMA200
        above = True
        if sma200:
            qb = qqq.loc[:day]
            if len(qb) >= 200:
                above = qb["close"].iloc[-1] > qb["close"].iloc[-200:].mean()

        if not above and hold and closes.get(hold[0]):
            cap += hold[2] * (closes[hold[0]] - hold[1])
            hold = None
            highest_price = 0
            tr += 1

        # Momentum ranking (with optional risk-adjust)
        mom = {}
        for s in POOL:
            before = all_raw[s][all_raw[s].index <= idx]
            if len(before) < 63:
                continue
            c = before["close"].values
            mom[s] = 0.5 * (c[-1] / c[-21] - 1) + 0.5 * (c[-1] / c[-63] - 1)
        cands = set(s for s, sc in sorted(mom.items(), key=lambda x: -x[1])[:2] if sc > 0)

        # Sell
        if hold and all_sell[hold[0]][idx] > 0 and closes.get(hold[0]):
            cap += hold[2] * (closes[hold[0]] - hold[1])
            hold = None
            highest_price = 0
            tr += 1

        # Buy with dynamic allocation (Task 1)
        if not hold and above:
            best = None
            for s in cands:
                if all_buy[s][idx] > 0 and closes.get(s):
                    if best is None or all_buy[s][idx] > all_buy[best][idx]:
                        best = s
            if best and closes[best]:
                alloc = 0.72
                q = int(cap * alloc / closes[best])
                if q > 0:
                    hold = (best, closes[best], q)
                    highest_price = closes[best]
                    tr += 1

        if hold and closes.get(hold[0]):
            eq.append(cap + hold[2] * (closes[hold[0]] - hold[1]))
        else:
            cap *= (1 + DAILY_YIELD)  # Cash yield (Task 6)
            eq.append(cap)

    return calc(eq, tr)


def pr(label, r):
    print(f"  {label:<42} {r['sharpe']:>7.3f} {r['cagr']:>7.1f}% "
          f"{r['maxdd']:>7.1f}% {r['trades']:>5} {r['win_rate']:>5.1f}% ${r['final']:>9,.0f}")


def main():
    print("=" * 85)
    print("  FUTU-QUANT Full System Backtest")
    print("=" * 85)

    print("\nPhase 1: Loading data & computing indicators...")
    all_raw, all_ind = {}, {}
    for sym in POOL:
        all_raw[sym] = load(sym)
        all_ind[sym] = add_indicators(all_raw[sym])
        print(f"  {sym}: {len(all_raw[sym])} bars ({all_raw[sym]['time_key'].iloc[0].date()} ~ {all_raw[sym]['time_key'].iloc[-1].date()})")

    print("\nPhase 2: Precomputing all strategy signals (one-time, ~2-5 min)...")
    all_buy, all_sell = {}, {}
    for sym in POOL:
        b, s = precompute_all_signals(sym, all_ind[sym])
        all_buy[sym] = b
        all_sell[sym] = s
        n_b = np.sum(b > 0)
        n_s = np.sum(s > 0)
        print(f"    {sym}: {n_b} BUY signals, {n_s} SELL signals over {len(b)} days")

    print("\nPhase 2b: Applying signal quality filter...")
    all_buy, all_sell = SIG_FILTER.filter_signals_vectorized(
        all_buy, all_sell, all_ind, POOL
    )
    for sym in POOL:
        n_b = np.sum(all_buy[sym] > 0)
        n_s = np.sum(all_sell[sym] > 0)
        print(f"    {sym} after filter: {n_b} BUY, {n_s} SELL")

    print("\nPhase 3: Running backtests...")
    segments = [
        ("10yr (full)", "2016-05-01", "2026-04-17"),
        ("5yr",         "2021-04-01", "2026-04-17"),
        ("3yr",         "2023-04-01", "2026-04-17"),
        ("1yr",         "2025-04-01", "2026-04-17"),
    ]

    hdr = (f"  {'Strategy':<42} {'Sharpe':>7} {'CAGR':>7} "
           f"{'MaxDD':>7} {'Trds':>5} {'WinR':>5} {'$3000->':>10}")
    sep = f"  {'-'*42} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*10}"

    for seg_name, start, end in segments:
        print(f"\n{'=' * 85}")
        print(f"  {seg_name}  ({start} ~ {end})")
        print(f"{'=' * 85}")

        # Buy & Hold
        print(f"\n  [Baseline] Buy & Hold")
        print(hdr); print(sep)
        for sym in POOL:
            df = all_raw[sym]
            m = (df["time_key"] >= start) & (df["time_key"] <= end)
            sub = df[m]
            if len(sub) > 20:
                p = sub["close"].values
                eq = [INITIAL_CAPITAL * x / p[0] for x in p]
                pr(f"{sym} buy&hold", calc(eq, 1))

        # Per-symbol strategies
        print(f"\n  [Per-Symbol] Multi-Strategy Trading")
        print(hdr); print(sep)
        for sym in POOL:
            r = bt_per_symbol(sym, all_raw[sym], all_buy[sym], all_sell[sym], start, end)
            pr(f"{sym} multi-strat", r)

        # Full system
        print(f"\n  [Full System] 4-ETF Rotation + Multi-Strategy")
        print(hdr); print(sep)
        r1 = bt_rotation(all_raw, all_buy, all_sell, start, end, sma200=False)
        r2 = bt_rotation(all_raw, all_buy, all_sell, start, end, sma200=True)
        pr("Rotation (no SMA200)", r1)
        pr("Rotation + SMA200 (LIVE config)", r2)

    print(f"\n{'=' * 85}")
    print("  $3000 initial. Per-symbol uses 95% allocation. Rotation uses 72%.")
    print("  Full system = daily momentum top-2 + strategy signals + SMA200 + hard stop -8%.")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
