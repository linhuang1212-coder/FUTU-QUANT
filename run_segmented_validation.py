"""Segmented validation framework: 10yr / 5yr / 3yr + stress tests + rolling WF.

Pass criteria (from plan):
  - 10yr/5yr/3yr Sharpe ALL > 1.0
  - Stress periods max drawdown < 40%
  - Rolling Walk-Forward (3yr train + 1yr test, 8 windows) consistency >= 60%

Usage:
    python run_segmented_validation.py                     # validate existing swing strategies
    python run_segmented_validation.py --mode intraday     # validate intraday strategies
    python run_segmented_validation.py --mode all          # both
"""

import sys, io, os, math, itertools, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data.downloader import load_daily, load_5min
from data.synthesizer import synthesize_intraday

# ── Time Segments ──────────────────────────────────────────────

SEGMENTS = {
    "10yr_full": ("2016-01-01", "2026-04-30"),
    "5yr_recent": ("2021-01-01", "2026-04-30"),
    "3yr_recent": ("2023-01-01", "2026-04-30"),
}

STRESS_PERIODS = {
    "COVID_crash": ("2020-01-01", "2020-06-30"),
    "rate_hike_bear": ("2022-01-01", "2022-12-31"),
    "AI_bull": ("2023-01-01", "2024-12-31"),
}

PASS_SHARPE = 1.0
STRESS_MAX_DD = 40.0
WF_CONSISTENCY = 0.60
WF_TRAIN_YEARS = 3
WF_TEST_YEARS = 1


# ── Data Loading ───────────────────────────────────────────────

def load_swing_data(symbols):
    data = {}
    for sym in symbols:
        ticker = sym.split(".")[-1] if "." in sym else sym
        df = load_daily(ticker)
        if df is not None and len(df) > 100:
            df["time_key"] = pd.to_datetime(df["time_key"])
            data[sym] = df
            print(f"  {sym}: {len(df)} daily bars ({df['time_key'].iloc[0].date()} ~ {df['time_key'].iloc[-1].date()})")
        else:
            print(f"  {sym}: no data")
    return data


def load_intraday_data(symbols, min_days=200):
    data = {}
    for sym in symbols:
        ticker = sym.split(".")[-1] if "." in sym else sym
        real = load_5min(ticker)
        daily = load_daily(ticker)
        if daily is not None and len(daily) >= min_days:
            synth = synthesize_intraday(daily)
            synth["time_key"] = pd.to_datetime(synth["time_key"])
            data[sym] = synth
            print(f"  {sym}: {len(synth)} synthetic 5min bars ({len(daily)} days)")
        elif real is not None:
            real["time_key"] = pd.to_datetime(real["time_key"])
            data[sym] = real
            print(f"  {sym}: {len(real)} real 5min bars")
        else:
            print(f"  {sym}: no data")
    return data


def slice_by_date(df, start, end):
    mask = (df["time_key"] >= start) & (df["time_key"] <= end)
    sub = df.loc[mask].reset_index(drop=True)
    return sub


# ── Backtest Engine (from run_param_scan.py) ───────────────────

def fast_backtest(signals, strengths, closes, initial_capital=3000.0,
                  commission_pct=0.001, slippage_pct=0.0005):
    capital = initial_capital
    position = 0
    avg_entry = 0.0
    trades = []
    equity = np.empty(len(closes))
    n = len(closes)

    for i in range(n):
        price = closes[i]
        sig = signals[i]

        if sig == 1 and position == 0:
            buy_p = price * (1 + slippage_pct)
            qty = int(capital * 0.95 / buy_p)
            if qty > 0:
                comm = buy_p * qty * commission_pct
                cost = buy_p * qty + comm
                if cost <= capital:
                    position = qty
                    avg_entry = buy_p
                    capital -= cost
                    trades.append({"type": "BUY", "price": buy_p, "qty": qty, "bar": i})

        elif sig == -1 and position > 0:
            sell_p = price * (1 - slippage_pct)
            comm = sell_p * position * commission_pct
            revenue = sell_p * position - comm
            pnl = (sell_p - avg_entry) * position - comm
            capital += revenue
            trades.append({"type": "SELL", "price": sell_p, "qty": position, "pnl": pnl, "bar": i})
            position = 0
            avg_entry = 0.0

        equity[i] = capital + position * price

    final = capital + position * closes[-1]
    return {"initial": initial_capital, "final": final, "trades": trades, "equity": equity}


def compute_metrics(result, closes):
    initial = result["initial"]
    final = result["final"]
    trades = result["trades"]
    equity = result["equity"]
    n = len(equity)

    ret_pct = (final / initial - 1) * 100
    years = max(n / 252, 0.01)
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if final > 0 else 0

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd)) * 100

    sell_trades = [t for t in trades if t["type"] == "SELL"]
    n_trades = len(sell_trades)
    if n_trades == 0:
        return {"sharpe": 0, "max_dd": max_dd, "return_pct": ret_pct, "cagr_pct": cagr,
                "trades": 0, "win_rate": 0, "profit_factor": 0}

    wins = [t["pnl"] for t in sell_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in sell_trades if t["pnl"] <= 0]
    win_rate = len(wins) / n_trades * 100
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0)

    rets = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    rf = 1.05 ** (1 / 252) - 1
    excess = rets - rf
    std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0
    sharpe = float(np.mean(excess) / std * math.sqrt(252)) if std > 1e-12 else 0

    return {
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 2),
        "return_pct": round(ret_pct, 2),
        "cagr_pct": round(cagr, 2),
        "trades": n_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else pf,
    }


# ── Indicator Precomputation ──────────────────────────────────

def precompute_indicators(df):
    out = df.copy()
    c = out["close"]
    for p in [2, 5, 8, 10, 14, 15, 20, 50, 200]:
        out[f"ma_{p}"] = c.rolling(p, min_periods=p).mean()
    for p in [5, 8, 10, 15, 20, 50]:
        out[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()
    for p in [2, 5, 7, 10, 14]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_g = gain.rolling(p).mean()
        avg_l = loss.rolling(p).mean()
        rs = avg_g / avg_l.replace(0, np.inf)
        out[f"rsi_{p}"] = 100 - (100 / (1 + rs))
    for bp in [15, 20]:
        for bs in [1.5, 2.0]:
            sma = c.rolling(bp).mean()
            std = c.rolling(bp).std()
            out[f"bb_{bp}_{bs}_upper"] = sma + bs * std
            out[f"bb_{bp}_{bs}_middle"] = sma
            out[f"bb_{bp}_{bs}_lower"] = sma - bs * std
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift()).abs(),
        (out["low"] - out["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["vol_ma20"] = out["volume"].rolling(20).mean()

    # Weekly returns for momentum rotation
    out["ret_1m"] = c.pct_change(21)
    out["ret_3m"] = c.pct_change(63)
    out["dayofweek"] = pd.to_datetime(out["time_key"]).dt.dayofweek
    out["daily_return"] = c.pct_change()
    return out


# ═══════════════════════════════════════════════════════════════
#  SWING SIGNAL GENERATORS
# ═══════════════════════════════════════════════════════════════

def gen_sma200_filter(df, params):
    """S1: QQQ > SMA200 -> hold TQQQ, else cash."""
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)
    close = df["close"].values
    ma200 = df["ma_200"].values

    in_pos = False
    for i in range(201, n):
        if np.isnan(ma200[i]):
            continue
        if not in_pos and close[i] > ma200[i]:
            signals[i] = 1
            strengths[i] = 70
            in_pos = True
        elif in_pos and close[i] < ma200[i]:
            signals[i] = -1
            strengths[i] = 70
            in_pos = False
    return signals, strengths


def gen_dual_ma_cross(df, params):
    """S2: EMA fast/slow crossover."""
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)
    fast_p = params.get("fast_ema", 10)
    slow_p = params.get("slow_ema", 50)
    fast = df[f"ema_{fast_p}"].values
    slow = df[f"ema_{slow_p}"].values

    in_pos = False
    for i in range(slow_p + 2, n):
        if np.isnan(fast[i]) or np.isnan(slow[i]):
            continue
        if not in_pos and fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]:
            signals[i] = 1
            strengths[i] = 70
            in_pos = True
        elif in_pos and (fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]):
            signals[i] = -1
            strengths[i] = 70
            in_pos = False
    return signals, strengths


def gen_momentum_rotation(df_dict, params):
    """S3: Weekly rotation between symbols. Returns dict of signals per symbol."""
    rebalance_day = params.get("rebalance_day", 4)  # Friday
    lookback = params.get("lookback", 21)

    all_signals = {}
    symbols = list(df_dict.keys())
    if len(symbols) < 2:
        return {s: (np.zeros(len(df_dict[s]), dtype=int), np.zeros(len(df_dict[s]))) for s in symbols}

    ref_sym = symbols[0]
    ref_dates = pd.to_datetime(df_dict[ref_sym]["time_key"]).values

    for sym in symbols:
        n = len(df_dict[sym])
        all_signals[sym] = (np.zeros(n, dtype=int), np.zeros(n, dtype=float))

    for sym in symbols:
        df = df_dict[sym]
        n = len(df)
        signals = np.zeros(n, dtype=int)
        strengths = np.zeros(n, dtype=float)
        close = df["close"].values
        dow = df["dayofweek"].values
        ret_col = df[f"ret_1m"].values

        in_pos = False
        for i in range(lookback + 5, n):
            if np.isnan(ret_col[i]):
                continue
            if dow[i] == rebalance_day:
                if ret_col[i] > 0 and not in_pos:
                    signals[i] = 1
                    strengths[i] = 60
                    in_pos = True
                elif ret_col[i] <= 0 and in_pos:
                    signals[i] = -1
                    strengths[i] = 60
                    in_pos = False
        all_signals[sym] = (signals, strengths)

    return all_signals


def gen_rsi2_reversion(df, params):
    """S4: RSI(2) < buy_thresh -> buy, RSI(2) > sell_thresh -> sell."""
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)
    buy_t = params.get("rsi_buy", 10)
    sell_t = params.get("rsi_sell", 90)
    rsi = df["rsi_2"].values

    in_pos = False
    for i in range(5, n):
        if np.isnan(rsi[i]):
            continue
        if not in_pos and rsi[i] < buy_t:
            signals[i] = 1
            strengths[i] = 80
            in_pos = True
        elif in_pos and rsi[i] > sell_t:
            signals[i] = -1
            strengths[i] = 80
            in_pos = False
    return signals, strengths


def gen_turnaround_tuesday(df, params):
    """S5: Monday drop > threshold -> buy Monday close, sell Tuesday close."""
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)
    drop_thresh = params.get("drop_pct", 1.0)  # in %
    close = df["close"].values
    dow = df["dayofweek"].values
    daily_ret = df["daily_return"].values

    for i in range(2, n):
        if np.isnan(daily_ret[i]):
            continue
        if dow[i] == 0 and daily_ret[i] * 100 < -drop_thresh:
            signals[i] = 1
            strengths[i] = 75
        elif dow[i] == 1:
            signals[i] = -1
            strengths[i] = 75
    return signals, strengths


# Existing strategies from run_param_scan.py
def gen_momentum_signals(df, params):
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)
    fast_col = f"ma_{params['fast_ma']}"
    slow_col = f"ma_{params['slow_ma']}"
    rsi_col = f"rsi_{params['rsi_period']}"
    fast_ema = f"ema_{params['fast_ma']}"
    slow_ema = f"ema_{params['slow_ma']}"
    fast_ma = df[fast_col].values
    slow_ma = df[slow_col].values
    rsi = df[rsi_col].values
    fast_e = df[fast_ema].values
    slow_e = df[slow_ema].values
    vol = df["volume"].values
    vol_ma = df["vol_ma20"].values
    vt = params["vol_threshold"]

    in_position = False
    for i in range(params["slow_ma"] + 5, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(rsi[i]):
            continue
        vol_ratio = (np.mean(vol[max(0, i-2):i+1]) / vol_ma[i]) if vol_ma[i] > 0 else 0
        if not in_position:
            cross = any(fast_ma[i-k-1] <= slow_ma[i-k-1] and fast_ma[i-k] > slow_ma[i-k]
                        for k in range(3) if i-k-1 >= 0)
            if cross and rsi[i] > 30 and vol_ratio >= vt:
                signals[i] = 1; strengths[i] = 70; in_position = True; continue
            if fast_e[i] > slow_e[i] and i > 0 and rsi[i-1] < 50 <= rsi[i]:
                signals[i] = 1; strengths[i] = 65; in_position = True; continue
        else:
            cross = any(fast_ma[i-k-1] >= slow_ma[i-k-1] and fast_ma[i-k] < slow_ma[i-k]
                        for k in range(3) if i-k-1 >= 0)
            if cross and rsi[i] < 70:
                signals[i] = -1; strengths[i] = 70; in_position = False
    return signals, strengths


def gen_mean_reversion_signals(df, params):
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)
    bp, bs = params["bb_period"], params["bb_std"]
    rsi_col = f"rsi_{params['rsi_period']}"
    lower = df[f"bb_{bp}_{bs}_lower"].values
    middle = df[f"bb_{bp}_{bs}_middle"].values
    upper = df[f"bb_{bp}_{bs}_upper"].values
    rsi = df[rsi_col].values
    close = df["close"].values
    os_t = params["rsi_oversold"]
    ob_t = params["rsi_overbought"]

    in_pos = False
    for i in range(bp + 5, n):
        if np.isnan(rsi[i]) or np.isnan(lower[i]):
            continue
        if not in_pos:
            if close[i] <= lower[i] or rsi[i] <= os_t:
                signals[i] = 1; strengths[i] = 70; in_pos = True
        else:
            if close[i] >= middle[i] or close[i] >= upper[i] or rsi[i] >= ob_t:
                signals[i] = -1; strengths[i] = 70; in_pos = False
    return signals, strengths


def gen_breakout_signals(df, params):
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)
    lb = params["lookback"]
    vt = params["vol_threshold"]
    atr_m = params["atr_mult"]
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values
    vol_ma = df["vol_ma20"].values
    atr = df["atr_14"].values

    in_pos = False
    for i in range(lb + 15, n):
        if np.isnan(atr[i]) or vol_ma[i] <= 0:
            continue
        vol_ratio = vol[i] / vol_ma[i]
        resistance = np.max(high[i-lb:i])
        support = np.min(low[i-lb:i])
        if not in_pos:
            if (close[i] > resistance or close[i] > close[i-1] + atr_m * atr[i]) and vol_ratio >= vt:
                signals[i] = 1; strengths[i] = 70; in_pos = True
        else:
            if close[i] < support or close[i] < close[i-1] - atr_m * atr[i]:
                signals[i] = -1; strengths[i] = 70; in_pos = False
    return signals, strengths


def gen_rsi_reversal_signals(df, params):
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)
    rsi_col = f"rsi_{params['rsi_period']}"
    rsi = df[rsi_col].values
    buy_t = params["rsi_buy"]
    sell_t = params["rsi_sell"]

    in_pos = False
    for i in range(params["rsi_period"] + 5, n):
        if np.isnan(rsi[i]):
            continue
        if not in_pos and rsi[i] <= buy_t:
            signals[i] = 1; strengths[i] = 70; in_pos = True
        elif in_pos and rsi[i] >= sell_t:
            signals[i] = -1; strengths[i] = 70; in_pos = False
    return signals, strengths


# ═══════════════════════════════════════════════════════════════
#  Strategy Registry
# ═══════════════════════════════════════════════════════════════

SWING_STRATEGIES = {
    "S1_SMA200_Filter": {
        "gen_fn": gen_sma200_filter,
        "grid": [{}],
    },
    "S2_Dual_MA_Cross": {
        "gen_fn": gen_dual_ma_cross,
        "grid": [
            {"fast_ema": 10, "slow_ema": 50},
            {"fast_ema": 8, "slow_ema": 50},
            {"fast_ema": 10, "slow_ema": 20},
            {"fast_ema": 5, "slow_ema": 20},
        ],
    },
    "S4_RSI2_Reversion": {
        "gen_fn": gen_rsi2_reversion,
        "grid": [
            {"rsi_buy": 5, "rsi_sell": 90},
            {"rsi_buy": 10, "rsi_sell": 90},
            {"rsi_buy": 5, "rsi_sell": 80},
            {"rsi_buy": 10, "rsi_sell": 80},
            {"rsi_buy": 15, "rsi_sell": 85},
        ],
    },
    "S5_Turnaround_Tue": {
        "gen_fn": gen_turnaround_tuesday,
        "grid": [
            {"drop_pct": 1.0},
            {"drop_pct": 1.5},
            {"drop_pct": 2.0},
            {"drop_pct": 0.5},
        ],
    },
    "Momentum": {
        "gen_fn": gen_momentum_signals,
        "grid": [
            {"fast_ma": 5, "slow_ma": 20, "rsi_period": 14, "vol_threshold": 1.0},
            {"fast_ma": 8, "slow_ma": 20, "rsi_period": 14, "vol_threshold": 1.0},
            {"fast_ma": 5, "slow_ma": 15, "rsi_period": 10, "vol_threshold": 1.3},
            {"fast_ma": 8, "slow_ma": 15, "rsi_period": 10, "vol_threshold": 1.0},
        ],
    },
    "Mean_Reversion": {
        "gen_fn": gen_mean_reversion_signals,
        "grid": [
            {"bb_period": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70},
            {"bb_period": 15, "bb_std": 1.5, "rsi_period": 10, "rsi_oversold": 25, "rsi_overbought": 75},
            {"bb_period": 20, "bb_std": 1.5, "rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 70},
        ],
    },
    "Breakout": {
        "gen_fn": gen_breakout_signals,
        "grid": [
            {"lookback": 20, "vol_threshold": 1.2, "atr_mult": 1.5},
            {"lookback": 10, "vol_threshold": 1.5, "atr_mult": 2.0},
            {"lookback": 20, "vol_threshold": 1.5, "atr_mult": 2.0},
        ],
    },
    "RSI_Reversal": {
        "gen_fn": gen_rsi_reversal_signals,
        "grid": [
            {"rsi_period": 5, "rsi_buy": 25, "rsi_sell": 75},
            {"rsi_period": 7, "rsi_buy": 30, "rsi_sell": 70},
            {"rsi_period": 5, "rsi_buy": 30, "rsi_sell": 70},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
#  Intraday Strategies (from run_intraday_scan.py)
# ═══════════════════════════════════════════════════════════════

def precompute_intraday(df):
    from data.indicators import TechnicalIndicators
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
    out["ma_200_daily"] = out["close"].rolling(200 * 78, min_periods=100 * 78).mean()
    return out


def split_into_days(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["time_key"]).dt.date
    days = []
    for _, group in df.groupby("date"):
        day_df = group.reset_index(drop=True)
        if len(day_df) >= 20:
            days.append(day_df)
    return days


def sim_rebalance_2pm_day(day_df, params):
    move_threshold_pct = params.get("move_threshold_pct", 2.0)
    entry_bar_idx = params.get("entry_bar_idx", 54)
    stop_pct = params.get("stop_pct", 1.5)

    if len(day_df) <= entry_bar_idx + 3:
        return None
    open_price = day_df["open"].iloc[0]
    price_at_entry = day_df["close"].iloc[entry_bar_idx]
    move_pct = (price_at_entry - open_price) / open_price * 100
    if move_pct < move_threshold_pct:
        return None
    entry_price = price_at_entry * 1.0005
    stop_price = entry_price * (1 - stop_pct / 100)
    for i in range(entry_bar_idx + 1, len(day_df)):
        if day_df["low"].iloc[i] <= stop_price:
            return (stop_price * 0.9995 - entry_price) / entry_price * 100
    eod_price = day_df["close"].iloc[-1] * 0.9995
    return (eod_price - entry_price) / entry_price * 100


def sim_vwap_trend_day(day_df, params):
    confirm_bars = params.get("confirm_bars", 3)
    rsi_floor = params.get("rsi_floor", 45)
    min_bars = params.get("min_bars_before_entry", 6)
    if len(day_df) < min_bars + confirm_bars + 5 or "vwap" not in day_df.columns:
        return None
    close = day_df["close"].values
    vwap = day_df["vwap"].values
    rsi = day_df["rsi_14"].values if "rsi_14" in day_df.columns else np.full(len(day_df), 50)
    position = 0
    entry_price = 0.0
    for i in range(min_bars, len(day_df)):
        if np.isnan(vwap[i]) or np.isnan(rsi[i]):
            continue
        if position == 0:
            above = sum(1 for k in range(confirm_bars) if i - k >= 0 and close[i - k] > vwap[i - k])
            if above >= confirm_bars and rsi[i] > rsi_floor:
                entry_price = close[i] * 1.0005
                position = 1
        else:
            if close[i] < vwap[i]:
                return (close[i] * 0.9995 - entry_price) / entry_price * 100
    if position > 0:
        return (day_df["close"].iloc[-1] * 0.9995 - entry_price) / entry_price * 100
    return None


def sim_sma200_intraday(day_df, params):
    """I1: Only trade VWAP trend when daily SMA200 filter is bullish."""
    if "ma_200_daily" not in day_df.columns:
        return sim_vwap_trend_day(day_df, params)
    ma200 = day_df["ma_200_daily"].iloc[0]
    if np.isnan(ma200):
        return None
    if day_df["close"].iloc[0] < ma200:
        return None
    return sim_vwap_trend_day(day_df, params)


def sim_afternoon_momentum(day_df, params):
    """I2: Afternoon momentum extension - flexible entry time and threshold."""
    return sim_rebalance_2pm_day(day_df, params)


def sim_volatility_squeeze(day_df, params):
    """I3: Volatility squeeze breakout."""
    squeeze_ratio = params.get("squeeze_ratio", 0.5)
    entry_after_bar = params.get("entry_after_bar", 24)
    stop_pct = params.get("stop_pct", 1.5)

    if len(day_df) < entry_after_bar + 10:
        return None

    first_2hr_high = day_df["high"].iloc[:entry_after_bar].max()
    first_2hr_low = day_df["low"].iloc[:entry_after_bar].min()
    first_2hr_range = first_2hr_high - first_2hr_low

    if first_2hr_range <= 0:
        return None

    avg_range = day_df["high"].mean() - day_df["low"].mean()
    if avg_range <= 0:
        return None

    if first_2hr_range / avg_range > squeeze_ratio * entry_after_bar:
        return None

    position = 0
    entry_price = 0.0
    stop_price = 0.0

    for i in range(entry_after_bar, len(day_df)):
        close = day_df["close"].iloc[i]
        if position == 0:
            if close > first_2hr_high:
                entry_price = close * 1.0005
                stop_price = entry_price * (1 - stop_pct / 100)
                position = 1
        else:
            if day_df["low"].iloc[i] <= stop_price:
                return (stop_price * 0.9995 - entry_price) / entry_price * 100

    if position > 0:
        return (day_df["close"].iloc[-1] * 0.9995 - entry_price) / entry_price * 100
    return None


INTRADAY_STRATEGIES = {
    "Rebalance_2pm": {
        "sim_fn": sim_rebalance_2pm_day,
        "grid": [
            {"move_threshold_pct": 1.5, "entry_bar_idx": 48, "stop_pct": 1.0},
            {"move_threshold_pct": 2.0, "entry_bar_idx": 54, "stop_pct": 1.5},
            {"move_threshold_pct": 1.5, "entry_bar_idx": 54, "stop_pct": 1.0},
            {"move_threshold_pct": 3.0, "entry_bar_idx": 54, "stop_pct": 2.0},
            {"move_threshold_pct": 1.0, "entry_bar_idx": 48, "stop_pct": 1.0},
        ],
    },
    "VWAP_Trend": {
        "sim_fn": sim_vwap_trend_day,
        "grid": [
            {"confirm_bars": 3, "rsi_floor": 50, "min_bars_before_entry": 6},
            {"confirm_bars": 2, "rsi_floor": 45, "min_bars_before_entry": 6},
            {"confirm_bars": 4, "rsi_floor": 50, "min_bars_before_entry": 9},
        ],
    },
    "I1_SMA200_Intraday": {
        "sim_fn": sim_sma200_intraday,
        "grid": [
            {"confirm_bars": 3, "rsi_floor": 50, "min_bars_before_entry": 6},
            {"confirm_bars": 2, "rsi_floor": 45, "min_bars_before_entry": 6},
        ],
    },
    "I2_Afternoon_Ext": {
        "sim_fn": sim_afternoon_momentum,
        "grid": [
            {"move_threshold_pct": 1.5, "entry_bar_idx": 36, "stop_pct": 1.0},
            {"move_threshold_pct": 1.5, "entry_bar_idx": 42, "stop_pct": 1.0},
            {"move_threshold_pct": 2.0, "entry_bar_idx": 48, "stop_pct": 1.5},
            {"move_threshold_pct": 1.0, "entry_bar_idx": 54, "stop_pct": 1.0},
            {"move_threshold_pct": 2.0, "entry_bar_idx": 60, "stop_pct": 1.5},
        ],
    },
    "I3_Vol_Squeeze": {
        "sim_fn": sim_volatility_squeeze,
        "grid": [
            {"squeeze_ratio": 0.3, "entry_after_bar": 24, "stop_pct": 1.5},
            {"squeeze_ratio": 0.5, "entry_after_bar": 24, "stop_pct": 1.0},
            {"squeeze_ratio": 0.5, "entry_after_bar": 18, "stop_pct": 1.5},
            {"squeeze_ratio": 0.4, "entry_after_bar": 24, "stop_pct": 2.0},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
#  Intraday Backtest Engine
# ═══════════════════════════════════════════════════════════════

def backtest_intraday(days, sim_fn, params, capital=3000.0):
    equity = capital
    trades = []
    alloc = 0.95
    equities = [capital]

    for day_df in days:
        pnl_pct = sim_fn(day_df, params)
        if pnl_pct is not None:
            dollar_pnl = equity * alloc * pnl_pct / 100
            equity += dollar_pnl
            trades.append(pnl_pct)
        equities.append(equity)

    equities = np.array(equities)
    total_ret = (equity / capital - 1) * 100
    n_trades = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0)

    if len(equities) > 1:
        rets = np.diff(equities) / np.where(equities[:-1] > 0, equities[:-1], 1)
        rf = 1.05 ** (1 / 252) - 1
        std = float(np.std(rets, ddof=1))
        sharpe = float(np.mean(rets - rf) / std * math.sqrt(252)) if std > 1e-12 else 0
        peak = np.maximum.accumulate(equities)
        dd = (peak - equities) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd)) * 100
    else:
        sharpe = 0
        max_dd = 0

    years = max(len(days) / 252, 0.01)
    cagr = ((equity / capital) ** (1 / years) - 1) * 100 if equity > 0 else 0

    return {
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 2),
        "return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "trades": n_trades,
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else pf,
    }


# ═══════════════════════════════════════════════════════════════
#  Segmented Validation Logic
# ═══════════════════════════════════════════════════════════════

def validate_swing_strategy(name, gen_fn, params, data_dict, symbols):
    results = {}

    for sym in symbols:
        if sym not in data_dict:
            continue
        full_df = data_dict[sym]

        sym_results = {"segments": {}, "stress": {}, "wf": {}}

        # 1. Segment tests
        for seg_name, (start, end) in SEGMENTS.items():
            seg_df = slice_by_date(full_df, start, end)
            if len(seg_df) < 50:
                sym_results["segments"][seg_name] = {"sharpe": 0, "max_dd": 0, "trades": 0, "return_pct": 0}
                continue
            seg_df = precompute_indicators(seg_df)
            sigs, strs = gen_fn(seg_df, params)
            bt = fast_backtest(sigs, strs, seg_df["close"].values)
            m = compute_metrics(bt, seg_df["close"].values)
            sym_results["segments"][seg_name] = m

        # 2. Stress tests
        for stress_name, (start, end) in STRESS_PERIODS.items():
            stress_df = slice_by_date(full_df, start, end)
            if len(stress_df) < 20:
                sym_results["stress"][stress_name] = {"sharpe": 0, "max_dd": 0, "trades": 0}
                continue
            stress_df = precompute_indicators(stress_df)
            sigs, strs = gen_fn(stress_df, params)
            bt = fast_backtest(sigs, strs, stress_df["close"].values)
            m = compute_metrics(bt, stress_df["close"].values)
            sym_results["stress"][stress_name] = m

        # 3. Rolling Walk-Forward
        dates = pd.to_datetime(full_df["time_key"])
        min_date = dates.min()
        max_date = dates.max()
        wf_wins = 0
        wf_total = 0

        train_start = min_date
        while True:
            train_end = train_start + pd.DateOffset(years=WF_TRAIN_YEARS)
            test_end = train_end + pd.DateOffset(years=WF_TEST_YEARS)
            if test_end > max_date:
                break

            train_df = slice_by_date(full_df, str(train_start.date()), str(train_end.date()))
            test_df = slice_by_date(full_df, str(train_end.date()), str(test_end.date()))

            if len(train_df) < 100 or len(test_df) < 50:
                train_start += pd.DateOffset(years=1)
                continue

            test_df = precompute_indicators(test_df)
            sigs, strs = gen_fn(test_df, params)
            bt = fast_backtest(sigs, strs, test_df["close"].values)
            m = compute_metrics(bt, test_df["close"].values)

            wf_total += 1
            if m["sharpe"] > 0:
                wf_wins += 1

            train_start += pd.DateOffset(years=1)

        wf_consistency = wf_wins / wf_total if wf_total > 0 else 0
        sym_results["wf"] = {"wins": wf_wins, "total": wf_total, "consistency": round(wf_consistency, 2)}

        results[sym] = sym_results

    return results


def validate_intraday_strategy(name, sim_fn, params, data_dict, symbols):
    results = {}

    for sym in symbols:
        if sym not in data_dict:
            continue
        full_df = data_dict[sym]
        full_df = precompute_intraday(full_df)

        sym_results = {"segments": {}, "stress": {}, "wf": {}}

        # 1. Segment tests
        for seg_name, (start, end) in SEGMENTS.items():
            seg_df = slice_by_date(full_df, start, end)
            if len(seg_df) < 78:
                sym_results["segments"][seg_name] = {"sharpe": 0, "max_dd": 0, "trades": 0, "return_pct": 0}
                continue
            days = split_into_days(seg_df)
            m = backtest_intraday(days, sim_fn, params)
            sym_results["segments"][seg_name] = m

        # 2. Stress tests
        for stress_name, (start, end) in STRESS_PERIODS.items():
            stress_df = slice_by_date(full_df, start, end)
            if len(stress_df) < 78:
                sym_results["stress"][stress_name] = {"sharpe": 0, "max_dd": 0, "trades": 0}
                continue
            days = split_into_days(stress_df)
            m = backtest_intraday(days, sim_fn, params)
            sym_results["stress"][stress_name] = m

        # 3. Rolling Walk-Forward
        dates = pd.to_datetime(full_df["time_key"])
        min_date = dates.min()
        max_date = dates.max()
        wf_wins = 0
        wf_total = 0

        train_start = min_date
        while True:
            train_end = train_start + pd.DateOffset(years=WF_TRAIN_YEARS)
            test_end = train_end + pd.DateOffset(years=WF_TEST_YEARS)
            if test_end > max_date:
                break

            test_df = slice_by_date(full_df, str(train_end.date()), str(test_end.date()))
            if len(test_df) < 78:
                train_start += pd.DateOffset(years=1)
                continue

            days = split_into_days(test_df)
            m = backtest_intraday(days, sim_fn, params)
            wf_total += 1
            if m["sharpe"] > 0:
                wf_wins += 1
            train_start += pd.DateOffset(years=1)

        wf_consistency = wf_wins / wf_total if wf_total > 0 else 0
        sym_results["wf"] = {"wins": wf_wins, "total": wf_total, "consistency": round(wf_consistency, 2)}

        results[sym] = sym_results

    return results


def judge(results):
    """Evaluate pass/fail for a strategy's validation results."""
    for sym, sr in results.items():
        segs = sr.get("segments", {})
        stress = sr.get("stress", {})
        wf = sr.get("wf", {})

        seg_pass = all(segs.get(s, {}).get("sharpe", 0) >= PASS_SHARPE
                       for s in SEGMENTS if s in segs and segs[s].get("trades", 0) > 0)

        stress_pass = all(stress.get(s, {}).get("max_dd", 100) < STRESS_MAX_DD
                          for s in STRESS_PERIODS if s in stress and stress[s].get("trades", 0) > 0)

        wf_pass = wf.get("consistency", 0) >= WF_CONSISTENCY

        sr["seg_pass"] = seg_pass
        sr["stress_pass"] = stress_pass
        sr["wf_pass"] = wf_pass
        sr["overall_pass"] = seg_pass and stress_pass and wf_pass

    return results


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Segmented Validation Framework")
    parser.add_argument("--mode", choices=["swing", "intraday", "all"], default="all")
    parser.add_argument("--symbols", nargs="+", default=["TQQQ", "SOXL"])
    parser.add_argument("--output", default="results/segmented_validation.csv")
    args = parser.parse_args()

    symbols_full = [f"US.{s}" if not s.startswith("US.") else s for s in args.symbols]
    tickers = [s.split(".")[-1] for s in symbols_full]

    print("=" * 80)
    print("FUTU-QUANT Segmented Validation Framework")
    print("=" * 80)
    print(f"Symbols:  {tickers}")
    print(f"Mode:     {args.mode}")
    print(f"Segments: {list(SEGMENTS.keys())}")
    print(f"Stress:   {list(STRESS_PERIODS.keys())}")
    print(f"Pass:     Sharpe>{PASS_SHARPE} (all segs), DD<{STRESS_MAX_DD}% (stress), WF>={WF_CONSISTENCY*100}%")
    print()

    all_rows = []

    if args.mode in ("swing", "all"):
        print("[SWING] Loading daily data...")
        swing_data = load_swing_data(symbols_full)

        print(f"\n[SWING] Validating {len(SWING_STRATEGIES)} strategies...\n")
        for strat_name, cfg in SWING_STRATEGIES.items():
            gen_fn = cfg["gen_fn"]
            for pi, params in enumerate(cfg["grid"]):
                label = f"{strat_name}#{pi}"
                print(f"  {label} params={params}")

                results = validate_swing_strategy(strat_name, gen_fn, params, swing_data, symbols_full)
                results = judge(results)

                for sym, sr in results.items():
                    ticker = sym.split(".")[-1]
                    segs = sr.get("segments", {})
                    wf = sr.get("wf", {})
                    row = {
                        "type": "swing",
                        "strategy": strat_name,
                        "param_idx": pi,
                        "params": str(params),
                        "symbol": ticker,
                        "10yr_sharpe": segs.get("10yr_full", {}).get("sharpe", 0),
                        "5yr_sharpe": segs.get("5yr_recent", {}).get("sharpe", 0),
                        "3yr_sharpe": segs.get("3yr_recent", {}).get("sharpe", 0),
                        "10yr_return": segs.get("10yr_full", {}).get("return_pct", 0),
                        "10yr_maxdd": segs.get("10yr_full", {}).get("max_dd", 0),
                        "10yr_trades": segs.get("10yr_full", {}).get("trades", 0),
                        "covid_dd": sr.get("stress", {}).get("COVID_crash", {}).get("max_dd", 0),
                        "rate_hike_dd": sr.get("stress", {}).get("rate_hike_bear", {}).get("max_dd", 0),
                        "wf_consistency": wf.get("consistency", 0),
                        "wf_windows": wf.get("total", 0),
                        "seg_pass": sr.get("seg_pass", False),
                        "stress_pass": sr.get("stress_pass", False),
                        "wf_pass": sr.get("wf_pass", False),
                        "PASS": sr.get("overall_pass", False),
                    }
                    all_rows.append(row)

                    tag = "PASS" if sr.get("overall_pass") else "FAIL"
                    sh10 = segs.get("10yr_full", {}).get("sharpe", 0)
                    sh5 = segs.get("5yr_recent", {}).get("sharpe", 0)
                    sh3 = segs.get("3yr_recent", {}).get("sharpe", 0)
                    wfc = wf.get("consistency", 0)
                    print(f"    {ticker}: 10y={sh10:.2f} 5y={sh5:.2f} 3y={sh3:.2f} WF={wfc:.0%} [{tag}]")

    if args.mode in ("intraday", "all"):
        print("\n[INTRADAY] Loading 5min data (synthesized from daily)...")
        intraday_data = load_intraday_data(symbols_full)

        print(f"\n[INTRADAY] Validating {len(INTRADAY_STRATEGIES)} strategies...\n")
        for strat_name, cfg in INTRADAY_STRATEGIES.items():
            sim_fn = cfg["sim_fn"]
            for pi, params in enumerate(cfg["grid"]):
                label = f"{strat_name}#{pi}"
                print(f"  {label} params={params}")

                results = validate_intraday_strategy(strat_name, sim_fn, params, intraday_data, symbols_full)
                results = judge(results)

                for sym, sr in results.items():
                    ticker = sym.split(".")[-1]
                    segs = sr.get("segments", {})
                    wf = sr.get("wf", {})
                    row = {
                        "type": "intraday",
                        "strategy": strat_name,
                        "param_idx": pi,
                        "params": str(params),
                        "symbol": ticker,
                        "10yr_sharpe": segs.get("10yr_full", {}).get("sharpe", 0),
                        "5yr_sharpe": segs.get("5yr_recent", {}).get("sharpe", 0),
                        "3yr_sharpe": segs.get("3yr_recent", {}).get("sharpe", 0),
                        "10yr_return": segs.get("10yr_full", {}).get("return_pct", 0),
                        "10yr_maxdd": segs.get("10yr_full", {}).get("max_dd", 0),
                        "10yr_trades": segs.get("10yr_full", {}).get("trades", 0),
                        "covid_dd": sr.get("stress", {}).get("COVID_crash", {}).get("max_dd", 0),
                        "rate_hike_dd": sr.get("stress", {}).get("rate_hike_bear", {}).get("max_dd", 0),
                        "wf_consistency": wf.get("consistency", 0),
                        "wf_windows": wf.get("total", 0),
                        "seg_pass": sr.get("seg_pass", False),
                        "stress_pass": sr.get("stress_pass", False),
                        "wf_pass": sr.get("wf_pass", False),
                        "PASS": sr.get("overall_pass", False),
                    }
                    all_rows.append(row)

                    tag = "PASS" if sr.get("overall_pass") else "FAIL"
                    sh10 = segs.get("10yr_full", {}).get("sharpe", 0)
                    sh5 = segs.get("5yr_recent", {}).get("sharpe", 0)
                    sh3 = segs.get("3yr_recent", {}).get("sharpe", 0)
                    wfc = wf.get("consistency", 0)
                    print(f"    {ticker}: 10y={sh10:.2f} 5y={sh5:.2f} 3y={sh3:.2f} WF={wfc:.0%} [{tag}]")

    # Summary
    if all_rows:
        df = pd.DataFrame(all_rows)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)

        passed = df[df["PASS"] == True]
        failed = df[df["PASS"] == False]

        print("\n" + "=" * 80)
        print(f"VALIDATION COMPLETE: {len(passed)} PASSED, {len(failed)} FAILED")
        print("=" * 80)

        if len(passed) > 0:
            print("\n  PASSED STRATEGIES:")
            for _, r in passed.iterrows():
                print(f"    [{r['type']:8}] {r['strategy']:25} @ {r['symbol']:6} "
                      f"10y={r['10yr_sharpe']:.2f} 5y={r['5yr_sharpe']:.2f} 3y={r['3yr_sharpe']:.2f} "
                      f"WF={r['wf_consistency']:.0%}")
                print(f"             params={r['params']}")

        print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
