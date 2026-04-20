"""Intraday strategy scan: ORB, VWAP Trend, First Pullback.

All strategies are TREND-FOLLOWING (no counter-trend), designed for
leveraged ETFs (TQQQ, SOXL) on 5-minute bars.

Key design changes from the failed v1 (RSI reversal):
  - Trend-following, not mean reversion
  - Wider stops (ATR-based instead of fixed %)
  - Let winners run (trailing stop, not fixed target)
  - Day-boundary-aware: each trading day is independent

Usage: python run_intraday_scan.py
"""

import sys, io, os, time, math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_saved_fd = os.dup(1)

from data.history import HistoryManager
from data.indicators import TechnicalIndicators
from data.downloader import load_5min, load_daily
from data.synthesizer import synthesize_intraday, BARS_PER_DAY as _SYNTH_BPD
from utils.helpers import load_yaml, get_project_root

sys.stdout = io.TextIOWrapper(os.fdopen(_saved_fd, "wb"), encoding="utf-8", errors="replace")

SYMBOLS = ["US.TQQQ", "US.SOXL"]
BARS_PER_DAY = 78  # 6.5 hours * 12 bars/hour (5-min)


def fetch_5min_data(symbols, lookback_days=120, min_days=200):
    """Load 5min data: local CSV -> Futu cache -> synthesize from daily.

    If real 5min data has fewer than min_days trading days,
    falls back to synthesizing from 10-year daily data.
    """
    result = {}

    for sym in symbols:
        ticker = sym.split(".")[-1] if "." in sym else sym

        # Priority 1: local real 5min CSV
        local = load_5min(ticker)
        if local is not None:
            n_days_real = len(set(pd.to_datetime(local["time_key"]).dt.date))
            if n_days_real >= min_days:
                result[sym] = local
                print(f"  {sym}: {len(local)} bars ({n_days_real} days, real 5min CSV)")
                continue

        # Priority 2: Futu cache
        hm = HistoryManager()
        cached = hm.load_from_cache(sym, "K_5M")
        if cached is not None:
            n_days_cached = len(set(pd.to_datetime(cached["time_key"]).dt.date))
            if n_days_cached >= min_days:
                result[sym] = cached
                print(f"  {sym}: {len(cached)} bars ({n_days_cached} days, Futu cache)")
                continue

        # Priority 3: synthesize from daily data (10 years)
        daily = load_daily(ticker)
        if daily is not None and len(daily) >= min_days:
            print(f"  {sym}: synthesizing from {len(daily)} daily bars...")
            synth = synthesize_intraday(daily)
            n_days_synth = len(daily)
            result[sym] = synth
            print(f"  {sym}: {len(synth)} bars ({n_days_synth} days, synthetic from daily)")
            continue

        # Priority 4: Futu API (last resort)
        try:
            root = get_project_root()
            settings = load_yaml(str(root / "config" / "settings.yaml"))
            from futu import OpenQuoteContext, RET_OK, KLType
            ctx = OpenQuoteContext(host=settings["futu"]["host"],
                                  port=settings["futu"]["port"])
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            all_pages = []
            page_key = None
            while True:
                kwargs = dict(code=sym, start=start_date, end=end_date,
                              ktype=KLType.K_5M, max_count=1000)
                if page_key is not None:
                    kwargs["page_req_key"] = page_key
                ret, data, page_key = ctx.request_history_kline(**kwargs)
                if ret == RET_OK and data is not None and len(data) > 0:
                    all_pages.append(data)
                else:
                    break
                if page_key is None:
                    break
                time.sleep(0.5)
            ctx.close()
            if all_pages:
                df = pd.concat(all_pages, ignore_index=True).drop_duplicates(
                    subset=["time_key"], keep="last"
                ).sort_values("time_key").reset_index(drop=True)
                hm.save_to_cache(sym, "K_5M", df)
                result[sym] = df
                print(f"  {sym}: {len(df)} bars (Futu API)")
        except Exception as e:
            print(f"  {sym}: no data available ({e})")

    return result


def split_into_days(df):
    """Split 5min DataFrame into list of per-day DataFrames."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["time_key"]).dt.date
    days = []
    for date, group in df.groupby("date"):
        day_df = group.reset_index(drop=True)
        if len(day_df) >= 20:
            days.append(day_df)
    return days


def precompute_intraday(df):
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


# ═══════════════════════════════════════════════════════════════
#  Strategy 1: Opening Range Breakout (ORB)
# ═══════════════════════════════════════════════════════════════

def sim_orb_day(day_df, params):
    """Simulate ORB on a single day. Returns PnL or None."""
    orb_bars = params.get("orb_bars", 6)  # 30 min = 6 x 5min bars
    vol_mult = params.get("vol_mult", 1.0)
    stop_type = params.get("stop_type", "mid")  # mid, low, atr
    target_mult = params.get("target_mult", 1.5)  # R:R ratio
    trail_atr = params.get("trail_atr", 2.0)

    if len(day_df) < orb_bars + 10:
        return None

    orb_high = day_df["high"].iloc[:orb_bars].max()
    orb_low = day_df["low"].iloc[:orb_bars].min()
    orb_range = orb_high - orb_low

    if orb_range <= 0 or np.isnan(orb_range):
        return None

    orb_mid = (orb_high + orb_low) / 2
    vol_avg = day_df["volume"].iloc[:orb_bars].mean()

    position = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    best_price = 0.0

    for i in range(orb_bars, len(day_df)):
        close = day_df["close"].iloc[i]
        high = day_df["high"].iloc[i]
        low = day_df["low"].iloc[i]
        vol = day_df["volume"].iloc[i]

        if position == 0:
            if close > orb_high and vol > vol_mult * vol_avg:
                entry_price = close * 1.0005  # slippage
                if stop_type == "mid":
                    stop_price = orb_mid
                elif stop_type == "low":
                    stop_price = orb_low
                else:
                    atr_val = day_df.get("atr_14")
                    if atr_val is not None and not pd.isna(atr_val.iloc[i]):
                        stop_price = entry_price - trail_atr * atr_val.iloc[i]
                    else:
                        stop_price = orb_mid

                risk = entry_price - stop_price
                if risk <= 0:
                    continue
                target_price = entry_price + target_mult * risk
                best_price = entry_price
                position = 1
        else:
            best_price = max(best_price, high)

            # Trailing stop: once profit > 1R, trail at entry
            risk = entry_price - stop_price if stop_price < entry_price else orb_range
            if best_price >= entry_price + risk:
                trail = max(stop_price, best_price - trail_atr * orb_range)
                stop_price = trail

            if low <= stop_price:
                sell_price = stop_price * 0.9995
                return (sell_price - entry_price) / entry_price * 100

            if high >= target_price:
                sell_price = target_price * 0.9995
                return (sell_price - entry_price) / entry_price * 100

    # EOD close
    if position > 0:
        eod_price = day_df["close"].iloc[-1] * 0.9995
        return (eod_price - entry_price) / entry_price * 100

    return None


# ═══════════════════════════════════════════════════════════════
#  Strategy 2: VWAP Trend Following
# ═══════════════════════════════════════════════════════════════

def sim_vwap_trend_day(day_df, params):
    """Buy when price establishes above VWAP, sell on VWAP break."""
    confirm_bars = params.get("confirm_bars", 3)
    rsi_floor = params.get("rsi_floor", 45)
    stop_atr_mult = params.get("stop_atr_mult", 1.5)
    min_bars_before_entry = params.get("min_bars_before_entry", 6)  # wait 30 min

    if len(day_df) < min_bars_before_entry + confirm_bars + 5:
        return None
    if "vwap" not in day_df.columns:
        return None

    close = day_df["close"].values
    vwap = day_df["vwap"].values
    rsi_col = "rsi_14"
    rsi = day_df[rsi_col].values if rsi_col in day_df.columns else np.full(len(day_df), 50)

    position = 0
    entry_price = 0.0

    for i in range(min_bars_before_entry, len(day_df)):
        if np.isnan(vwap[i]) or np.isnan(rsi[i]):
            continue

        if position == 0:
            above_count = 0
            for k in range(confirm_bars):
                idx = i - k
                if idx >= 0 and close[idx] > vwap[idx]:
                    above_count += 1
            if above_count >= confirm_bars and rsi[i] > rsi_floor:
                entry_price = close[i] * 1.0005
                position = 1
        else:
            # Exit: close below VWAP
            if close[i] < vwap[i]:
                sell_price = close[i] * 0.9995
                return (sell_price - entry_price) / entry_price * 100

            # Hard stop: down more than stop_atr_mult * day's range so far
            day_range = day_df["high"].iloc[:i+1].max() - day_df["low"].iloc[:i+1].min()
            if day_range > 0 and (entry_price - close[i]) / day_range > stop_atr_mult:
                sell_price = close[i] * 0.9995
                return (sell_price - entry_price) / entry_price * 100

    # EOD close
    if position > 0:
        eod_price = day_df["close"].iloc[-1] * 0.9995
        return (eod_price - entry_price) / entry_price * 100

    return None


# ═══════════════════════════════════════════════════════════════
#  Strategy 3: First Pullback to EMA
# ═══════════════════════════════════════════════════════════════

def sim_first_pullback_day(day_df, params):
    """Buy on first pullback to EMA after initial rally, if above VWAP."""
    ema_col = f"ema_{params.get('ema_period', 8)}"
    pullback_pct = params.get("pullback_pct", 0.3)
    rsi_floor = params.get("rsi_floor", 40)
    min_rally_pct = params.get("min_rally_pct", 0.5)

    if ema_col not in day_df.columns or "vwap" not in day_df.columns:
        return None
    if len(day_df) < 20:
        return None

    close = day_df["close"].values
    high = day_df["high"].values
    low = day_df["low"].values
    ema = day_df[ema_col].values
    vwap = day_df["vwap"].values
    rsi = day_df["rsi_14"].values if "rsi_14" in day_df.columns else np.full(len(day_df), 50)

    opening_price = close[0]
    day_high = opening_price
    found_rally = False
    position = 0
    entry_price = 0.0

    for i in range(6, len(day_df)):
        if np.isnan(ema[i]) or np.isnan(vwap[i]) or np.isnan(rsi[i]):
            continue

        day_high = max(day_high, high[i])
        rally_pct = (day_high - opening_price) / opening_price * 100

        if not found_rally and rally_pct >= min_rally_pct:
            found_rally = True

        if position == 0 and found_rally:
            # Pullback: close near EMA and above VWAP
            dist_to_ema = abs(close[i] - ema[i]) / ema[i] * 100
            if (dist_to_ema < pullback_pct
                    and close[i] > vwap[i]
                    and close[i] < day_high * 0.995
                    and rsi[i] > rsi_floor):
                entry_price = close[i] * 1.0005
                position = 1
        elif position > 0:
            # Exit: close below VWAP
            if close[i] < vwap[i]:
                sell_price = close[i] * 0.9995
                return (sell_price - entry_price) / entry_price * 100

            # New high profit taking (optional, controlled by param)
            if close[i] > day_high * 0.998 and (close[i] - entry_price) / entry_price > 0.015:
                sell_price = close[i] * 0.9995
                return (sell_price - entry_price) / entry_price * 100

    # EOD close
    if position > 0:
        eod_price = day_df["close"].iloc[-1] * 0.9995
        return (eod_price - entry_price) / entry_price * 100

    return None


# ═══════════════════════════════════════════════════════════════
#  Strategy 4: Overnight Gap Mean Reversion
# ═══════════════════════════════════════════════════════════════

def sim_gap_reversion_day(day_df, params):
    """Mean reversion on overnight gap within first 30 minutes."""
    gap_threshold_pct = params.get("gap_threshold_pct", 1.5)
    entry_delay_bars = params.get("entry_delay_bars", 2)  # wait 10 min after open
    exit_bars = params.get("exit_bars", 6)  # close within 30 min of entry
    stop_pct = params.get("stop_pct", 2.0)

    if len(day_df) < exit_bars + entry_delay_bars + 5:
        return None

    # Gap = open price vs previous close (use first bar's open vs day_df close from yesterday)
    # Since we only have today's data, use first bar open vs first bar close to detect gap direction
    # Better: check if open is significantly different from prior close
    # We approximate: if first bar shows strong direction, fade it
    open_price = day_df["open"].iloc[0]

    # Use first few bars to establish gap direction
    # If open was up strongly (gap up), look for short-term reversal back down
    # Since we can only go LONG, we only trade gap-DOWN (oversold open -> bounce)

    # Check if price opens below previous day's level using VWAP as anchor
    vwap_col = "vwap"
    if vwap_col not in day_df.columns:
        return None

    # Gap down detection: first bar's close is significantly below its open
    # AND price is below VWAP in first few bars
    early_bars = day_df.iloc[:entry_delay_bars]
    gap_down = (open_price - early_bars["close"].iloc[-1]) / open_price * 100

    if gap_down < gap_threshold_pct:
        return None  # No significant gap down

    # Entry: after entry_delay_bars, if price shows sign of recovery
    entry_idx = entry_delay_bars
    if entry_idx >= len(day_df):
        return None

    entry_bar = day_df.iloc[entry_idx]
    # Require: current close > previous bar close (first sign of bounce)
    if entry_idx > 0 and entry_bar["close"] <= day_df["close"].iloc[entry_idx - 1]:
        return None

    entry_price = entry_bar["close"] * 1.0005  # slippage
    stop_price = entry_price * (1 - stop_pct / 100)

    # Hold for exit_bars, then close
    max_exit_idx = min(entry_idx + exit_bars, len(day_df) - 1)

    for i in range(entry_idx + 1, max_exit_idx + 1):
        low = day_df["low"].iloc[i]
        close = day_df["close"].iloc[i]

        if low <= stop_price:
            return (stop_price * 0.9995 - entry_price) / entry_price * 100

        if i == max_exit_idx:
            return (close * 0.9995 - entry_price) / entry_price * 100

    return None


# ═══════════════════════════════════════════════════════════════
#  Strategy 5: Leveraged ETF Rebalancing Effect (2pm entry)
# ═══════════════════════════════════════════════════════════════

def sim_rebalance_2pm_day(day_df, params):
    """Exploit leveraged ETF rebalancing: enter at 2pm on big move days."""
    move_threshold_pct = params.get("move_threshold_pct", 2.0)
    entry_bar_idx = params.get("entry_bar_idx", 54)  # ~2pm = bar 54 (9:30 + 4.5hrs = 54 5-min bars)
    stop_pct = params.get("stop_pct", 1.5)

    if len(day_df) <= entry_bar_idx + 3:
        return None

    open_price = day_df["open"].iloc[0]
    price_at_2pm = day_df["close"].iloc[entry_bar_idx]

    move_pct = (price_at_2pm - open_price) / open_price * 100

    # Only go long on big up days (sponsors will be buying into close)
    if move_pct < move_threshold_pct:
        return None

    entry_price = price_at_2pm * 1.0005  # slippage
    stop_price = entry_price * (1 - stop_pct / 100)

    # Hold from 2pm to close
    for i in range(entry_bar_idx + 1, len(day_df)):
        if day_df["low"].iloc[i] <= stop_price:
            return (stop_price * 0.9995 - entry_price) / entry_price * 100

    # EOD close
    eod_price = day_df["close"].iloc[-1] * 0.9995
    return (eod_price - entry_price) / entry_price * 100


# ═══════════════════════════════════════════════════════════════
#  Backtest engine for day-by-day strategies
# ═══════════════════════════════════════════════════════════════

def backtest_intraday_strategy(days, sim_fn, params, capital=3000.0):
    """Run a day-level strategy across all trading days."""
    equity = capital
    trades = []
    alloc = 0.95

    for day_df in days:
        pnl_pct = sim_fn(day_df, params)
        if pnl_pct is not None:
            dollar_pnl = equity * alloc * pnl_pct / 100
            equity += dollar_pnl
            trades.append(pnl_pct)

    total_ret = (equity / capital - 1) * 100
    n_trades = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0)

    n_days = len(days)
    daily_rets = []
    eq = capital
    trade_idx = 0
    for day_df in days:
        pnl_pct = sim_fn(day_df, params)
        if pnl_pct is not None:
            dr = eq * alloc * pnl_pct / 100
            eq += dr
        daily_rets.append(eq)

    daily_rets = np.array(daily_rets)
    if len(daily_rets) > 1:
        rets = np.diff(daily_rets) / np.where(daily_rets[:-1] > 0, daily_rets[:-1], 1)
        rf = (1.05 ** (1 / 252) - 1)
        std = float(np.std(rets, ddof=1))
        sharpe = float(np.mean(rets - rf) / std * math.sqrt(252)) if std > 1e-12 else 0
        peak = np.maximum.accumulate(daily_rets)
        dd = (peak - daily_rets) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd)) * 100
    else:
        sharpe = 0
        max_dd = 0

    return {
        "return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 2),
        "trades": n_trades,
        "trades_per_day": round(n_trades / max(n_days, 1), 2),
        "win_rate": round(wr, 1),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else pf,
        "n_days": n_days,
    }


# ═══════════════════════════════════════════════════════════════
#  Parameter grids
# ═══════════════════════════════════════════════════════════════

ORB_GRID = {
    "orb_bars": [3, 6, 9],        # 15min, 30min, 45min opening range
    "vol_mult": [0.8, 1.0, 1.2],
    "stop_type": ["mid", "low"],
    "target_mult": [1.5, 2.0, 3.0],
}

VWAP_GRID = {
    "confirm_bars": [2, 3, 4],
    "rsi_floor": [40, 45, 50],
    "stop_atr_mult": [1.0, 1.5, 2.0],
    "min_bars_before_entry": [3, 6, 9],
}

PULLBACK_GRID = {
    "ema_period": [8, 13],
    "pullback_pct": [0.2, 0.3, 0.5],
    "rsi_floor": [35, 40, 50],
    "min_rally_pct": [0.3, 0.5, 1.0],
}

GAP_GRID = {
    "gap_threshold_pct": [1.0, 1.5, 2.0],
    "entry_delay_bars": [1, 2, 3],
    "exit_bars": [4, 6, 9],
    "stop_pct": [1.5, 2.0, 3.0],
}

REBALANCE_GRID = {
    "move_threshold_pct": [1.5, 2.0, 3.0, 4.0],
    "entry_bar_idx": [48, 54, 60],  # ~1:30pm, ~2pm, ~2:30pm
    "stop_pct": [1.0, 1.5, 2.0],
}


def grid_search(strategy_name, sim_fn, grid, days_dict):
    import itertools
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"\n  {strategy_name}: {len(combos)} param combos")

    all_results = []
    for sym, days in days_dict.items():
        print(f"    {sym} ({len(days)} days)...", end=" ", flush=True)
        best = None
        sym_results = []

        for combo in combos:
            params = dict(zip(keys, combo))
            result = backtest_intraday_strategy(days, sim_fn, params)
            result["params"] = params
            result["symbol"] = sym
            result["strategy"] = strategy_name
            sym_results.append(result)

            if result["trades"] >= 3:
                if best is None or result["sharpe"] > best["sharpe"]:
                    best = result

        all_results.extend(sym_results)
        if best:
            print(f"Sharpe={best['sharpe']:.4f} Ret={best['return_pct']:+.1f}% "
                  f"Tr={best['trades']} WR={best['win_rate']:.0f}% PF={best['profit_factor']}")
            print(f"      Best params: {best['params']}")
        else:
            print("no valid results")

    return all_results


def walkforward_intraday(strategy_name, sim_fn, params, days, train_pct=0.65):
    """Simple walk-forward: train on first N days, validate on rest."""
    split = int(len(days) * train_pct)
    if split < 10 or len(days) - split < 5:
        return None, None
    train_days = days[:split]
    val_days = days[split:]
    train_r = backtest_intraday_strategy(train_days, sim_fn, params)
    val_r = backtest_intraday_strategy(val_days, sim_fn, params)
    return train_r, val_r


def main():
    print("=" * 70)
    print("INTRADAY STRATEGY SCAN v2 (Trend-Following)")
    print("=" * 70)

    print("\n[1/4] Fetching 5-min data...")
    all_data = fetch_5min_data(SYMBOLS, lookback_days=120)
    if not all_data:
        print("ERROR: No 5min data. Need FutuOpenD running.")
        return

    print("\n[2/4] Splitting into trading days & computing indicators...")
    days_dict = {}
    for sym, df in all_data.items():
        df = precompute_intraday(df)
        days = split_into_days(df)
        days_dict[sym] = days
        print(f"  {sym}: {len(days)} trading days, {len(df)} total bars")

    print("\n[3/4] Grid search (all strategies)...")
    all_results = []
    all_results.extend(grid_search("ORB", sim_orb_day, ORB_GRID, days_dict))
    all_results.extend(grid_search("VWAP_Trend", sim_vwap_trend_day, VWAP_GRID, days_dict))
    all_results.extend(grid_search("First_Pullback", sim_first_pullback_day, PULLBACK_GRID, days_dict))
    all_results.extend(grid_search("Gap_Reversion", sim_gap_reversion_day, GAP_GRID, days_dict))
    all_results.extend(grid_search("Rebalance_2pm", sim_rebalance_2pm_day, REBALANCE_GRID, days_dict))

    # Filter promising
    good = [r for r in all_results if r["sharpe"] > 0.2 and r["trades"] >= 3]
    good.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n\n{'=' * 70}")
    print(f"PROMISING INTRADAY COMBOS (Sharpe > 0.2, trades >= 3): {len(good)}")
    print("=" * 70)

    for r in good[:20]:
        print(f"  {r['strategy']:18} @ {r['symbol']:10} "
              f"Sh={r['sharpe']:.4f} Ret={r['return_pct']:+.1f}% "
              f"DD={r['max_dd']:.1f}% Tr={r['trades']} ({r['trades_per_day']:.2f}/day) "
              f"WR={r['win_rate']:.0f}% PF={r['profit_factor']}")
        print(f"    {r['params']}")

    # [4/4] Walk-forward on top combos
    print(f"\n\n{'=' * 70}")
    print("[4/4] Walk-Forward Validation on Top Combos")
    print("=" * 70)

    sim_fns = {
        "ORB": sim_orb_day,
        "VWAP_Trend": sim_vwap_trend_day,
        "First_Pullback": sim_first_pullback_day,
        "Gap_Reversion": sim_gap_reversion_day,
        "Rebalance_2pm": sim_rebalance_2pm_day,
    }
    validated = []

    for r in good[:10]:
        sym = r["symbol"]
        days = days_dict[sym]
        fn = sim_fns[r["strategy"]]
        train_r, val_r = walkforward_intraday(r["strategy"], fn, r["params"], days)

        if train_r is None or val_r is None:
            continue

        overfit = (1 - val_r["sharpe"] / train_r["sharpe"]) * 100 if train_r["sharpe"] != 0 else None
        verdict = "PASS" if val_r["sharpe"] > 0 and val_r["trades"] >= 2 else "FAIL"

        r["train_sharpe"] = train_r["sharpe"]
        r["val_sharpe"] = val_r["sharpe"]
        r["val_return"] = val_r["return_pct"]
        r["val_trades"] = val_r["trades"]
        r["val_wr"] = val_r["win_rate"]
        r["overfit"] = overfit
        r["verdict"] = verdict

        tag = "***" if verdict == "PASS" else "   "
        of_str = f"{overfit:+.0f}%" if overfit is not None else "N/A"
        print(f"  {tag} {r['strategy']:18}@{r['symbol']}: "
              f"Train_Sh={train_r['sharpe']:.4f} Val_Sh={val_r['sharpe']:.4f} "
              f"Val_Ret={val_r['return_pct']:+.1f}% Val_Tr={val_r['trades']} "
              f"OF={of_str} [{verdict}]")

        if verdict == "PASS":
            validated.append(r)

    print(f"\n\n{'=' * 70}")
    print(f"VALIDATED INTRADAY STRATEGIES: {len(validated)}")
    print("=" * 70)

    if validated:
        for r in validated:
            print(f"\n  {r['strategy']}@{r['symbol']}")
            print(f"    Full Sharpe:     {r['sharpe']:.4f}")
            print(f"    Val Sharpe:      {r['val_sharpe']:.4f}")
            print(f"    Val Return:      {r['val_return']:+.1f}%")
            print(f"    Trades/day:      {r['trades_per_day']:.2f}")
            print(f"    Win Rate:        {r['win_rate']:.0f}%")
            print(f"    Profit Factor:   {r['profit_factor']}")
            print(f"    Params:          {r['params']}")
    else:
        print("  No intraday strategies passed walk-forward validation.")
        print("  Data may be insufficient (need more 5min history) or strategies need more tuning.")

    Path("results").mkdir(exist_ok=True)
    if good:
        rows = []
        for r in good:
            flat = {k: v for k, v in r.items() if k != "params"}
            flat.update({f"p_{k}": v for k, v in r["params"].items()})
            rows.append(flat)
        pd.DataFrame(rows).to_csv("results/intraday_scan.csv", index=False)
        print(f"\nResults -> results/intraday_scan.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
