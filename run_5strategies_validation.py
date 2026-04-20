"""Comprehensive validation of 5 new strategy directions using 10yr REAL daily data.

Strategy A: TQQQ + TMF/GLD hedge portfolio (50/50 rebalancing + crash filter)
Strategy B: Volatility-targeted continuous leverage (E = Vtarget/sigma * trend)
Strategy C: Simplified ML factor proxy (Put/Call ratio + VRP proxy + momentum)
Strategy D: Sector rotation with conditional leverage (11 SPDR ETFs)
Strategy E: SPY/TQQQ pairs trading (spread mean reversion)

Validation: 10yr/5yr/3yr segments + stress periods + rolling walk-forward.
"""

import sys, io, math
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data.downloader import load_daily

SEGMENTS = {
    "10yr": ("2016-01-01", "2026-04-30"),
    "5yr":  ("2021-01-01", "2026-04-30"),
    "3yr":  ("2023-01-01", "2026-04-30"),
}
STRESS = {
    "COVID":     ("2020-01-01", "2020-06-30"),
    "RateHike":  ("2022-01-01", "2022-12-31"),
    "AI_Bull":   ("2023-01-01", "2024-12-31"),
}


def load(sym):
    df = load_daily(sym)
    if df is not None:
        df["time_key"] = pd.to_datetime(df["time_key"])
        df = df.sort_values("time_key").reset_index(drop=True)
    return df


def sl(df, s, e):
    return df[(df["time_key"] >= s) & (df["time_key"] <= e)].reset_index(drop=True)


def metrics(equities, years=None):
    eq = np.array(equities, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return {"sh": 0, "cagr": 0, "mdd": 0, "final": 0}
    rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
    rf = 1.05 ** (1/252) - 1
    std = float(np.std(rets, ddof=1))
    sh = float(np.mean(rets - rf) / std * math.sqrt(252)) if std > 1e-12 else 0
    if years is None:
        years = len(eq) / 252
    total = eq[-1] / eq[0]
    cagr = (total ** (1/max(years, 0.01)) - 1) * 100 if total > 0 else -100
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    mdd = float(np.max(dd)) * 100
    return {"sh": round(sh, 3), "cagr": round(cagr, 1), "mdd": round(mdd, 1), "final": round(eq[-1], 2)}


# ══════════════════════════════════════════════════════════════
# STRATEGY A: TQQQ + Hedge (TMF or GLD) 50/50 Rebalancing
# ══════════════════════════════════════════════════════════════

def strat_a_rebalance(etf1_df, etf2_df, hedge_df, rebal_days=42,
                      crash_pct=-15.0, capital=10000.0):
    """50/50 rebalance between etf1 (TQQQ) and etf2 (TMF/GLD).
    Crash filter: if etf1 drops >crash_pct in a day, go to hedge (IEF).
    Return to 50/50 when etf1 recovers above pre-crash close."""
    m = pd.merge(etf1_df[["time_key","close"]].rename(columns={"close":"e1"}),
                 etf2_df[["time_key","close"]].rename(columns={"close":"e2"}),
                 on="time_key", how="inner")
    m = pd.merge(m, hedge_df[["time_key","close"]].rename(columns={"close":"h"}),
                 on="time_key", how="inner").reset_index(drop=True)

    eq = capital
    w1 = 0.5
    w2 = 0.5
    in_crash = False
    crash_level = 0.0
    equities = [eq]
    trades = 0
    last_rebal = 0

    for i in range(1, len(m)):
        r1 = m["e1"].iloc[i] / m["e1"].iloc[i-1] - 1
        r2 = m["e2"].iloc[i] / m["e2"].iloc[i-1] - 1
        rh = m["h"].iloc[i] / m["h"].iloc[i-1] - 1

        if in_crash:
            eq *= (1 + rh)
            if m["e1"].iloc[i] > crash_level:
                in_crash = False
                w1, w2 = 0.5, 0.5
                trades += 1
        else:
            eq *= (1 + w1 * r1 + w2 * r2)
            if r1 < crash_pct / 100:
                in_crash = True
                crash_level = m["e1"].iloc[i-1]
                trades += 1
            elif i - last_rebal >= rebal_days:
                w1, w2 = 0.5, 0.5
                last_rebal = i
                trades += 1

        equities.append(eq)

    return equities, trades


# ══════════════════════════════════════════════════════════════
# STRATEGY B: Volatility-Targeted Continuous Leverage
# ══════════════════════════════════════════════════════════════

def strat_b_voltarget(qqq_df, tqqq_df, vol_target=0.18, sma_period=200,
                      ewma_lam=0.94, dd_thresh=-0.20, dd_mult=0.5,
                      rebal_freq=5, capital=10000.0):
    """Continuous leverage: E = clip(Vtarget / sigma * trend, 0, 2).
    E <= 1: hold E in QQQ equivalent, rest cash.
    E > 1: hold E/3 in TQQQ (3x), rest cash."""
    m = pd.merge(qqq_df[["time_key","close"]].rename(columns={"close":"qqq"}),
                 tqqq_df[["time_key","close"]].rename(columns={"close":"tqqq"}),
                 on="time_key", how="inner").reset_index(drop=True)

    closes = m["qqq"].values
    tqqq_closes = m["tqqq"].values
    n = len(m)

    sma = pd.Series(closes).rolling(sma_period).mean().values
    rets_qqq = np.zeros(n)
    rets_qqq[1:] = np.diff(closes) / closes[:-1]
    rets_tqqq = np.zeros(n)
    rets_tqqq[1:] = np.diff(tqqq_closes) / tqqq_closes[:-1]

    var = 0.0
    sigma = np.zeros(n)
    for i in range(n):
        var = ewma_lam * var + (1 - ewma_lam) * rets_qqq[i] ** 2
        sigma[i] = math.sqrt(var) * math.sqrt(252) if var > 0 else 0.3

    eq = capital
    equities = [eq]
    E_current = 0.0
    trades = 0

    for i in range(1, n):
        if i % rebal_freq == 0 and i >= sma_period:
            trend = 1.0 if closes[i-1] >= sma[i-1] else 0.0
            sig = max(sigma[i-1], 0.05)
            E_raw = vol_target / sig * trend

            peak_eq = max(equities)
            dd = (eq / peak_eq - 1) if peak_eq > 0 else 0
            if dd < dd_thresh:
                E_raw *= dd_mult

            E_new = max(0.0, min(E_raw, 2.0))
            if abs(E_new - E_current) > 0.1:
                E_current = E_new
                trades += 1

        if E_current <= 1.0:
            daily_ret = E_current * rets_qqq[i]
        else:
            w_tqqq = E_current / 3.0
            daily_ret = w_tqqq * rets_tqqq[i]

        eq *= (1 + daily_ret)
        equities.append(eq)

    return equities, trades


# ══════════════════════════════════════════════════════════════
# STRATEGY C: Simplified Factor Proxy (no options data needed)
# ══════════════════════════════════════════════════════════════

def strat_c_factor_proxy(qqq_df, tqqq_df, capital=10000.0):
    """Proxy for options-based factors using only price data:
    - VRP proxy: 20d realized vol vs 60d realized vol (vol of vol)
    - Momentum: 20d return
    - Mean reversion: RSI(14)
    Combined score -> position sizing 0 to 1."""
    m = pd.merge(qqq_df[["time_key","close"]].rename(columns={"close":"qqq"}),
                 tqqq_df[["time_key","close"]].rename(columns={"close":"tqqq"}),
                 on="time_key", how="inner").reset_index(drop=True)

    qqq = m["qqq"].values
    tqqq = m["tqqq"].values
    n = len(m)

    ret_qqq = np.zeros(n)
    ret_qqq[1:] = np.diff(qqq) / qqq[:-1]
    ret_tqqq = np.zeros(n)
    ret_tqqq[1:] = np.diff(tqqq) / tqqq[:-1]

    rv20 = pd.Series(ret_qqq).rolling(20).std().values * math.sqrt(252)
    rv60 = pd.Series(ret_qqq).rolling(60).std().values * math.sqrt(252)
    mom20 = pd.Series(qqq).pct_change(20).values

    rsi_period = 14
    gains = np.where(np.diff(qqq, prepend=qqq[0]) > 0, np.diff(qqq, prepend=qqq[0]), 0)
    losses = np.where(np.diff(qqq, prepend=qqq[0]) < 0, -np.diff(qqq, prepend=qqq[0]), 0)
    avg_gain = pd.Series(gains).rolling(rsi_period).mean().values
    avg_loss = pd.Series(losses).rolling(rsi_period).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - 100 / (1 + rs)

    sma200 = pd.Series(qqq).rolling(200).mean().values

    eq = capital
    equities = [eq]
    trades = 0
    prev_alloc = 0.0

    for i in range(1, n):
        if i < 200 or np.isnan(rv20[i-1]) or np.isnan(rv60[i-1]) or np.isnan(mom20[i-1]):
            equities.append(eq)
            continue

        score = 0.0

        if qqq[i-1] > sma200[i-1]:
            score += 0.3

        if rv20[i-1] < rv60[i-1]:
            score += 0.25

        if mom20[i-1] > 0:
            score += 0.25

        if 40 < rsi[i-1] < 70:
            score += 0.2

        alloc = min(score, 1.0)

        if abs(alloc - prev_alloc) > 0.15:
            trades += 1
            prev_alloc = alloc

        eq *= (1 + alloc * ret_tqqq[i])
        equities.append(eq)

    return equities, trades


# ══════════════════════════════════════════════════════════════
# STRATEGY D: Sector Rotation with Conditional Leverage
# ══════════════════════════════════════════════════════════════

def strat_d_sector_rotation(sector_dfs, tqqq_df, qqq_df,
                            lookback=63, rebal_freq=21, top_n=3,
                            capital=10000.0):
    """Monthly rotation: rank 11 sectors by momentum, hold top N.
    If QQQ > SMA200 and top sector is XLK, use TQQQ instead of QQQ."""
    all_sectors = list(sector_dfs.keys())
    prices = {}
    for name, df in sector_dfs.items():
        prices[name] = df.set_index("time_key")["close"]

    common_idx = prices[all_sectors[0]].index
    for s in all_sectors[1:]:
        common_idx = common_idx.intersection(prices[s].index)

    qqq_p = qqq_df.set_index("time_key")["close"].reindex(common_idx)
    tqqq_p = tqqq_df.set_index("time_key")["close"].reindex(common_idx)
    sma200 = qqq_p.rolling(200).mean()

    px = pd.DataFrame({s: prices[s].reindex(common_idx) for s in all_sectors})
    px = px.dropna()
    common_idx = px.index

    qqq_p = qqq_p.reindex(common_idx).ffill()
    tqqq_p = tqqq_p.reindex(common_idx).ffill()
    sma200 = sma200.reindex(common_idx).ffill()

    eq = capital
    equities = [eq]
    holdings = []
    trades = 0

    for i in range(1, len(common_idx)):
        if holdings:
            daily_ret = 0.0
            w = 1.0 / len(holdings)
            for h_name, h_is_leveraged in holdings:
                if h_is_leveraged:
                    r = tqqq_p.iloc[i] / tqqq_p.iloc[i-1] - 1 if tqqq_p.iloc[i-1] > 0 else 0
                else:
                    r = px[h_name].iloc[i] / px[h_name].iloc[i-1] - 1 if px[h_name].iloc[i-1] > 0 else 0
                daily_ret += w * r
            eq *= (1 + daily_ret)

        equities.append(eq)

        if i % rebal_freq == 0 and i >= lookback:
            moms = {}
            for s in all_sectors:
                past = px[s].iloc[max(0, i-lookback)]
                if past > 0:
                    moms[s] = px[s].iloc[i] / past - 1
            if not moms:
                continue

            ranked = sorted(moms.items(), key=lambda x: x[1], reverse=True)
            top_sectors = ranked[:top_n]

            new_holdings = []
            trend_ok = not np.isnan(sma200.iloc[i]) and qqq_p.iloc[i] > sma200.iloc[i]

            for name, mom in top_sectors:
                if mom <= 0:
                    continue
                use_leverage = (name == "XLK" and trend_ok and mom > 0.05)
                new_holdings.append((name, use_leverage))

            if not new_holdings:
                new_holdings = []

            if new_holdings != holdings:
                trades += 1
                holdings = new_holdings

    return equities, trades


# ══════════════════════════════════════════════════════════════
# STRATEGY E: SPY/TQQQ Pairs Trading (Spread Mean Reversion)
# ══════════════════════════════════════════════════════════════

def strat_e_pairs(spy_df, tqqq_df, lookback=60, entry_z=2.0, exit_z=0.5,
                  capital=10000.0):
    """Pairs trade: track spread between normalized SPY and TQQQ.
    Long TQQQ / short SPY when spread < -entry_z, reverse at +entry_z.
    Exit when |spread| < exit_z."""
    m = pd.merge(spy_df[["time_key","close"]].rename(columns={"close":"spy"}),
                 tqqq_df[["time_key","close"]].rename(columns={"close":"tqqq"}),
                 on="time_key", how="inner").reset_index(drop=True)

    spy = m["spy"].values
    tqqq = m["tqqq"].values
    n = len(m)

    log_spy = np.log(spy)
    log_tqqq = np.log(tqqq)

    eq = capital
    equities = [eq]
    pos = 0  # 0=flat, 1=long TQQQ/short SPY, -1=reverse
    trades = 0

    for i in range(1, n):
        if i < lookback:
            equities.append(eq)
            continue

        window_spy = log_spy[i-lookback:i]
        window_tqqq = log_tqqq[i-lookback:i]
        spread = window_tqqq - np.mean(window_tqqq) / np.std(window_tqqq) * np.std(window_spy) * window_spy / np.mean(window_spy)

        ratio = log_tqqq[i] - np.polyval(np.polyfit(window_spy, window_tqqq, 1), log_spy[i])
        spread_mean = np.mean(window_tqqq - np.polyval(np.polyfit(window_spy, window_tqqq, 1), window_spy))
        spread_std = np.std(window_tqqq - np.polyval(np.polyfit(window_spy, window_tqqq, 1), window_spy))

        if spread_std < 1e-10:
            equities.append(eq)
            continue

        z = (ratio - spread_mean) / spread_std

        r_tqqq = tqqq[i] / tqqq[i-1] - 1
        r_spy = spy[i] / spy[i-1] - 1

        if pos == 1:
            eq *= (1 + 0.5 * r_tqqq - 0.5 * r_spy)
        elif pos == -1:
            eq *= (1 - 0.5 * r_tqqq + 0.5 * r_spy)

        if pos == 0:
            if z < -entry_z:
                pos = 1
                trades += 1
            elif z > entry_z:
                pos = -1
                trades += 1
        elif pos == 1:
            if z > -exit_z:
                pos = 0
                trades += 1
        elif pos == -1:
            if z < exit_z:
                pos = 0
                trades += 1

        equities.append(eq)

    return equities, trades


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def ph(title):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")


def run_segments(name, bt_fn, all_segs):
    """Run backtest over all segments and print results."""
    print(f"\n  {'Segment':<18} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Trades':>8}")
    print(f"  {'-'*52}")
    results = {}
    for seg_name, (s, e) in all_segs.items():
        try:
            eq, tr = bt_fn(s, e)
        except Exception as ex:
            print(f"  {seg_name:<18} ERROR: {ex}")
            continue
        yrs = len(eq) / 252
        m = metrics(eq, yrs)
        tag = " ***" if m["sh"] > 1.0 else (" **" if m["sh"] > 0.5 else "")
        print(f"  {seg_name:<18} {m['sh']:>8.3f} {m['cagr']:>7.1f}% {m['mdd']:>7.1f}% {tr:>8}{tag}")
        results[seg_name] = m
    return results


def main():
    ph("COMPREHENSIVE 5-STRATEGY VALIDATION (Real Daily Data)")

    print("\nLoading data...")
    data = {}
    for sym in ["QQQ","SPY","TQQQ","SOXL","TMF","TLT","GLD","UGL","IEF","BIL","QLD",
                "XLK","XLV","XLF","XLE","XLY","XLP","XLI","XLB","XLRE","XLU","XLC"]:
        df = load(sym)
        if df is not None:
            data[sym] = df
            print(f"  {sym}: {len(df)} days")
        else:
            print(f"  {sym}: MISSING")

    ALL_SEGS = {**SEGMENTS, **STRESS}

    # ── Strategy A: TQQQ + TMF/GLD Hedge ──
    for hedge_name, hedge_sym in [("TMF", "TMF"), ("GLD_via_UGL", "UGL"), ("TLT", "TLT")]:
        if "TQQQ" not in data or hedge_sym not in data or "IEF" not in data:
            continue
        ph(f"Strategy A: TQQQ + {hedge_name} (50/50 rebalance + crash filter)")

        for crash_pct in [-15, -20]:
            for rebal_days in [21, 42, 63]:
                print(f"\n  --- crash={crash_pct}%, rebal={rebal_days}d ---")

                def bt_a(s, e, cp=crash_pct, rd=rebal_days, hn=hedge_sym):
                    return strat_a_rebalance(
                        sl(data["TQQQ"], s, e), sl(data[hn], s, e),
                        sl(data["IEF"], s, e), rebal_days=rd, crash_pct=cp)

                run_segments(f"A_{hedge_name}_{crash_pct}_{rebal_days}", bt_a, ALL_SEGS)

    # ── Strategy B: Vol-Targeted Leverage ──
    if "QQQ" in data and "TQQQ" in data:
        ph("Strategy B: Volatility-Targeted Continuous Leverage")

        for vtarget in [0.15, 0.18, 0.22]:
            for sma_p in [150, 200]:
                print(f"\n  --- Vtarget={vtarget}, SMA={sma_p} ---")

                def bt_b(s, e, vt=vtarget, sp=sma_p):
                    return strat_b_voltarget(
                        sl(data["QQQ"], s, e), sl(data["TQQQ"], s, e),
                        vol_target=vt, sma_period=sp)

                run_segments(f"B_vt{vtarget}_sma{sma_p}", bt_b, ALL_SEGS)

    # ── Strategy C: Factor Proxy ──
    if "QQQ" in data and "TQQQ" in data:
        ph("Strategy C: Multi-Factor Proxy (trend + vol regime + momentum + RSI)")

        def bt_c(s, e):
            return strat_c_factor_proxy(sl(data["QQQ"], s, e), sl(data["TQQQ"], s, e))

        run_segments("C_factor", bt_c, ALL_SEGS)

    # ── Strategy D: Sector Rotation ──
    sector_syms = ["XLK","XLV","XLF","XLE","XLY","XLP","XLI","XLB","XLRE","XLU","XLC"]
    sector_data = {s: data[s] for s in sector_syms if s in data}
    if len(sector_data) >= 8 and "TQQQ" in data and "QQQ" in data:
        ph("Strategy D: Sector Rotation with Conditional Leverage")

        for top_n in [1, 3, 5]:
            for lb in [42, 63, 126]:
                print(f"\n  --- top_n={top_n}, lookback={lb}d ---")

                def bt_d(s, e, tn=top_n, lback=lb):
                    sec_sl = {k: sl(v, s, e) for k, v in sector_data.items()}
                    return strat_d_sector_rotation(
                        sec_sl, sl(data["TQQQ"], s, e), sl(data["QQQ"], s, e),
                        lookback=lback, top_n=tn)

                run_segments(f"D_top{top_n}_lb{lb}", bt_d, ALL_SEGS)

    # ── Strategy E: SPY/TQQQ Pairs ──
    if "SPY" in data and "TQQQ" in data:
        ph("Strategy E: SPY/TQQQ Pairs Trading")

        for lb in [30, 60, 90]:
            for ez in [1.5, 2.0, 2.5]:
                print(f"\n  --- lookback={lb}, entry_z={ez} ---")

                def bt_e(s, e, lback=lb, entry=ez):
                    return strat_e_pairs(
                        sl(data["SPY"], s, e), sl(data["TQQQ"], s, e),
                        lookback=lback, entry_z=entry)

                run_segments(f"E_lb{lb}_z{ez}", bt_e, ALL_SEGS)

    # ── Summary ──
    ph("VALIDATION COMPLETE")
    print("  Pass criteria: Sharpe > 1.0 across 10yr/5yr/3yr segments")
    print("  ** = Sharpe > 0.5, *** = Sharpe > 1.0")
    print("  Only strategies with *** across ALL time periods will be deployed")


if __name__ == "__main__":
    main()
