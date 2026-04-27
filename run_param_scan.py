"""High-performance parameter scan pipeline.

Pre-computes all technical indicators once per (symbol, param_set),
then runs a fast vectorized-ish backtest loop.

Usage:
    python run_param_scan.py
    python run_param_scan.py --output results/scan_results.csv
"""

import sys
import io
import argparse
import itertools
import math
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from data.history import HistoryManager
from data.indicators import TechnicalIndicators
from data.downloader import load_daily
from utils.helpers import load_yaml, get_project_root

SYMBOLS = ["US.TQQQ", "US.SOXL", "US.TNA", "US.QQQ", "US.SPY"]

STRATEGY_GRIDS = {
    "momentum": {
        "grid": {
            "fast_ma": [5, 8],
            "slow_ma": [15, 20],
            "rsi_period": [10, 14],
            "vol_threshold": [1.0, 1.3],
        },
    },
    "mean_reversion": {
        "grid": {
            "bb_period": [15, 20],
            "bb_std": [1.5, 2.0],
            "rsi_period": [10, 14],
            "rsi_oversold": [25, 30],
            "rsi_overbought": [70, 75],
        },
    },
    "breakout": {
        "grid": {
            "lookback": [10, 20],
            "vol_threshold": [1.2, 1.5],
            "atr_mult": [1.5, 2.0],
        },
    },
    "rsi_reversal": {
        "grid": {
            "rsi_period": [5, 7],
            "rsi_buy": [25, 30],
            "rsi_sell": [70, 75],
        },
    },
}


def fetch_all_data(symbols: list[str], start: str = "2015-01-01") -> dict[str, pd.DataFrame]:
    """Load daily data: local CSV first, then Futu cache, then Futu API."""
    root = get_project_root()
    settings = load_yaml(str(root / "config" / "settings.yaml"))
    hm = HistoryManager()
    result = {}

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = start

    ctx = None
    connected = False

    for sym in symbols:
        # Priority 1: local downloaded CSV
        ticker = sym.split(".")[-1] if "." in sym else sym
        local = load_daily(ticker)
        if local is not None and len(local) >= 500:
            result[sym] = local
            print(f"  {sym}: {len(local)} bars (local CSV)")
            continue

        # Priority 2: Futu cache
        cached = hm.load_from_cache(sym, "K_DAY")
        if cached is not None and len(cached) >= 2000:
            result[sym] = cached
            print(f"  {sym}: {len(cached)} bars (Futu cache)")
            continue

        # Priority 3: Futu API
        if not connected:
            try:
                from futu import OpenQuoteContext, RET_OK, KLType
                ctx = OpenQuoteContext(host=settings["futu"]["host"],
                                      port=settings["futu"]["port"])
                connected = True
            except Exception as e:
                print(f"[WARN] FutuOpenD unavailable ({e})")

        if connected and ctx is not None:
            import time as _time
            _time.sleep(0.3)
            try:
                from futu import RET_OK, KLType
                all_pages = []
                page_key = None
                while True:
                    kwargs = dict(code=sym, start=start_date, end=end_date,
                                  ktype=KLType.K_DAY, max_count=1000)
                    if page_key is not None:
                        kwargs["page_req_key"] = page_key
                    ret, data, page_key = ctx.request_history_kline(**kwargs)
                    if ret == RET_OK and data is not None and len(data) > 0:
                        all_pages.append(data)
                    else:
                        break
                    if page_key is None:
                        break
                    _time.sleep(0.3)

                if all_pages:
                    df = pd.concat(all_pages, ignore_index=True).drop_duplicates(
                        subset=["time_key"], keep="last"
                    ).sort_values("time_key").reset_index(drop=True)
                    hm.save_to_cache(sym, "K_DAY", df)
                    result[sym] = df
                    print(f"  {sym}: {len(df)} bars (Futu API, {len(all_pages)} pages)")
                else:
                    print(f"  {sym}: insufficient data")
            except Exception as e:
                print(f"  {sym}: error ({e})")
        else:
            print(f"  {sym}: no data")

    if ctx is not None:
        ctx.close()
    return result


def precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns we'll need across all strategies."""
    out = df.copy()
    for p in [5, 8, 10, 14, 15, 20]:
        col = f"ma_{p}"
        if col not in out.columns:
            out[col] = out["close"].rolling(p).mean()
    for p in [5, 8, 15, 20]:
        col = f"ema_{p}"
        if col not in out.columns:
            out[col] = out["close"].ewm(span=p, adjust=False).mean()
    for p in [5, 7, 10, 14]:
        col = f"rsi_{p}"
        if col not in out.columns:
            delta = out["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_g = gain.rolling(p).mean()
            avg_l = loss.rolling(p).mean()
            rs = avg_g / avg_l.replace(0, np.inf)
            out[col] = 100 - (100 / (1 + rs))
    for bp in [15, 20]:
        for bs in [1.5, 2.0]:
            tag = f"bb_{bp}_{bs}"
            if f"{tag}_upper" not in out.columns:
                sma = out["close"].rolling(bp).mean()
                std = out["close"].rolling(bp).std()
                out[f"{tag}_upper"] = sma + bs * std
                out[f"{tag}_middle"] = sma
                out[f"{tag}_lower"] = sma - bs * std
    for p in [14]:
        col = f"atr_{p}"
        if col not in out.columns:
            out = TechnicalIndicators.add_atr(out, p)
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def fast_backtest(
    signals: np.ndarray,
    strengths: np.ndarray,
    closes: np.ndarray,
    initial_capital: float = 3000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> dict:
    """Ultra-fast backtest. signals: 1=BUY, -1=SELL, 0=HOLD."""
    capital = initial_capital
    position = 0
    avg_entry = 0.0
    trades = []
    equity = np.empty(len(closes))
    n = len(closes)

    for i in range(n):
        price = closes[i]
        sig = signals[i]
        strength = strengths[i]

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
                    trades.append({"type": "BUY", "price": buy_p, "qty": qty, "comm": comm, "bar": i})

        elif sig == -1 and position > 0:
            sell_p = price * (1 - slippage_pct)
            comm = sell_p * position * commission_pct
            revenue = sell_p * position - comm
            pnl = (sell_p - avg_entry) * position - comm
            capital += revenue
            trades.append({"type": "SELL", "price": sell_p, "qty": position, "comm": comm, "pnl": pnl, "bar": i})
            position = 0
            avg_entry = 0.0

        equity[i] = capital + position * price

    final = capital + position * closes[-1]
    return {"initial": initial_capital, "final": final, "trades": trades, "equity": equity}


def compute_metrics(result: dict, closes: np.ndarray) -> dict:
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
        return {"total_return_pct": ret_pct, "cagr_pct": cagr, "max_drawdown_pct": max_dd,
                "sharpe_ratio": 0, "total_trades": 0, "win_rate_pct": 0, "profit_factor": 0,
                "calmar_ratio": 0, "sortino_ratio": 0}

    wins = [t["pnl"] for t in sell_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in sell_trades if t["pnl"] <= 0]
    win_rate = len(wins) / n_trades * 100
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0)

    rets = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    rf_per = (1.05 ** (1 / 252) - 1)
    excess = rets - rf_per

    std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0
    sharpe = float(np.mean(excess) / std * math.sqrt(252)) if std > 1e-12 else 0

    downside = np.minimum(0, excess)
    ddev = float(np.sqrt(np.mean(downside ** 2)))
    sortino = float(np.mean(excess) / ddev * math.sqrt(252)) if ddev > 1e-12 else 0

    calmar = (cagr / 100) / (max_dd / 100) if max_dd > 0.01 else 0

    bh_ret = (closes[-1] / closes[0] - 1) * 100

    return {
        "total_return_pct": round(ret_pct, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "total_trades": n_trades,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else pf,
        "buy_hold_return_pct": round(bh_ret, 2),
        "vs_benchmark_pct": round(ret_pct - bh_ret, 2),
    }


def gen_momentum_signals(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
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
    close = df["close"].values
    vt = params["vol_threshold"]

    in_position = False
    for i in range(params["slow_ma"] + 5, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(rsi[i]):
            continue

        vol_ratio = (np.mean(vol[max(0, i-2):i+1]) / vol_ma[i]) if vol_ma[i] > 0 else 0

        if not in_position:
            cross = False
            for k in range(3):
                ci, pi = i - k, i - k - 1
                if pi >= 0 and fast_ma[pi] <= slow_ma[pi] and fast_ma[ci] > slow_ma[ci]:
                    cross = True; break

            if cross and rsi[i] > 30 and vol_ratio >= vt:
                signals[i] = 1
                strengths[i] = min(50 + vol_ratio * 10 + (50 - abs(rsi[i] - 50)) * 0.5, 100)
                in_position = True
                continue

            if fast_e[i] > slow_e[i] and i > 0 and rsi[i-1] < 50 <= rsi[i]:
                signals[i] = 1
                strengths[i] = min(55 + (rsi[i] - 50) * 0.8, 95)
                in_position = True
                continue
        else:
            cross = False
            for k in range(3):
                ci, pi = i - k, i - k - 1
                if pi >= 0 and fast_ma[pi] >= slow_ma[pi] and fast_ma[ci] < slow_ma[ci]:
                    cross = True; break
            if cross and rsi[i] < 70:
                signals[i] = -1
                strengths[i] = 70
                in_position = False

    return signals, strengths


def gen_mean_reversion_signals(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)

    bp, bs = params["bb_period"], params["bb_std"]
    rsi_col = f"rsi_{params['rsi_period']}"
    upper = df[f"bb_{bp}_{bs}_upper"].values
    lower = df[f"bb_{bp}_{bs}_lower"].values
    middle = df[f"bb_{bp}_{bs}_middle"].values
    rsi = df[rsi_col].values
    close = df["close"].values
    os_thresh = params["rsi_oversold"]
    ob_thresh = params["rsi_overbought"]

    in_position = False
    for i in range(bp + 5, n):
        if np.isnan(rsi[i]) or np.isnan(lower[i]):
            continue
        if not in_position:
            bb_low = close[i] <= lower[i]
            rsi_low = rsi[i] <= os_thresh
            if bb_low and rsi_low:
                signals[i] = 1
                strengths[i] = min(70 + (os_thresh - rsi[i]) * 2, 95)
                in_position = True
            elif bb_low:
                signals[i] = 1
                strengths[i] = 55
                in_position = True
            elif rsi_low:
                signals[i] = 1
                strengths[i] = 55
                in_position = True
        else:
            if close[i] >= middle[i]:
                signals[i] = -1
                strengths[i] = 65
                in_position = False
            elif close[i] >= upper[i] or rsi[i] >= ob_thresh:
                signals[i] = -1
                strengths[i] = 75
                in_position = False

    return signals, strengths


def gen_breakout_signals(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
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

    in_position = False
    for i in range(lb + 15, n):
        if np.isnan(atr[i]) or vol_ma[i] <= 0:
            continue
        vol_ratio = vol[i] / vol_ma[i]
        resistance = np.max(high[i-lb:i])
        support = np.min(low[i-lb:i])

        if not in_position:
            price_break = close[i] > resistance
            atr_break = close[i] > close[i-1] + atr_m * atr[i] if i > 0 and np.isfinite(atr[i]) else False
            vol_ok = vol_ratio >= vt

            if (price_break or atr_break) and vol_ok:
                base = 55 if vol_ratio < 1.8 else 75
                strength = min(base + vol_ratio * 5, 95)
                signals[i] = 1
                strengths[i] = strength
                in_position = True
        else:
            if close[i] < support or (i > 0 and close[i] < close[i-1] - atr_m * atr[i]):
                signals[i] = -1
                strengths[i] = 70
                in_position = False

    return signals, strengths


def gen_rsi_reversal_signals(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)

    rsi_col = f"rsi_{params['rsi_period']}"
    rsi = df[rsi_col].values
    buy_t = params["rsi_buy"]
    sell_t = params["rsi_sell"]

    in_position = False
    for i in range(params["rsi_period"] + 5, n):
        if np.isnan(rsi[i]):
            continue
        if not in_position:
            if rsi[i] <= buy_t:
                signals[i] = 1
                strengths[i] = min(50 + (buy_t - rsi[i]) * 2, 90)
                in_position = True
        else:
            if rsi[i] >= sell_t:
                signals[i] = -1
                strengths[i] = min(50 + (rsi[i] - sell_t) * 2, 90)
                in_position = False
            elif i > 0 and rsi[i-1] < 50 <= rsi[i]:
                signals[i] = -1
                strengths[i] = 60
                in_position = False

    return signals, strengths


SIGNAL_GENERATORS = {
    "momentum": gen_momentum_signals,
    "mean_reversion": gen_mean_reversion_signals,
    "breakout": gen_breakout_signals,
    "rsi_reversal": gen_rsi_reversal_signals,
}


def walk_forward_scan(
    strat_name: str,
    df: pd.DataFrame,
    symbol: str,
    param_grid: dict,
    train_pct: float = 0.7,
) -> list[dict]:
    gen_fn = SIGNAL_GENERATORS[strat_name]
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    split = int(len(df) * train_pct)
    train_df = df.iloc[:split]
    val_df = df.iloc[split:].reset_index(drop=True)
    train_closes = train_df["close"].values
    val_closes = val_df["close"].values

    train_results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        sigs, strs = gen_fn(train_df, params)
        result = fast_backtest(sigs, strs, train_closes)
        metrics = compute_metrics(result, train_closes)
        if metrics["total_trades"] >= 2:
            train_results.append({"params": params, "metrics": metrics})

    if not train_results:
        return []

    train_results.sort(key=lambda x: x["metrics"].get("sharpe_ratio", 0), reverse=True)

    val_results = []
    for rank, entry in enumerate(train_results[:10], 1):
        params = entry["params"]
        sigs, strs = gen_fn(val_df, params)
        result = fast_backtest(sigs, strs, val_closes)
        metrics = compute_metrics(result, val_closes)

        train_sharpe = entry["metrics"]["sharpe_ratio"]
        val_sharpe = metrics["sharpe_ratio"]
        overfit = round((1 - val_sharpe / train_sharpe) * 100, 1) if train_sharpe and train_sharpe != 0 else None

        row = {
            "strategy": strat_name,
            "symbol": symbol,
            "train_rank": rank,
            "train_sharpe": train_sharpe,
            "train_return_pct": entry["metrics"]["total_return_pct"],
            "overfit_pct": overfit,
            **{f"p_{k}": v for k, v in params.items()},
            **metrics,
        }
        val_results.append(row)

    return val_results


# ── 并行 worker（顶层函数，Windows multiprocessing 兼容）────────────────────
import os as _os
from concurrent.futures import ProcessPoolExecutor as _PPE, as_completed as _asc


def _scan_worker(args):
    """单个 (strat, symbol) walk-forward 任务，在子进程中运行。"""
    strat_name, df_records, symbol, grid, train_pct = args
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import pandas as _pd
    df = _pd.DataFrame(df_records)
    results = walk_forward_scan(strat_name, df, symbol, grid, train_pct)
    return strat_name, symbol, results


def run_scan(output_path: str, n_jobs: int = -1):
    n_workers = _os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    print("=" * 60)
    print(f"FUTU-QUANT Parameter Scan  [workers={n_workers}]")
    print("=" * 60)

    print("\n[1/3] Fetching historical data...")
    all_data = fetch_all_data(SYMBOLS)
    if not all_data:
        print("ERROR: No data available.")
        return

    print("\n[2/3] Pre-computing indicators...")
    precomputed = {}
    for sym, df in all_data.items():
        precomputed[sym] = precompute_indicators(df)
        print(f"  {sym}: {len(df)} bars, {len(precomputed[sym].columns)} columns")

    # 构造并行任务
    tasks = []
    for strat_name, cfg in STRATEGY_GRIDS.items():
        grid = cfg["grid"]
        for sym, df in precomputed.items():
            # 序列化 DataFrame 为 records（子进程 pickle 友好）
            tasks.append((strat_name, df.to_dict("records"), sym, grid, 0.7))

    total_tasks = len(tasks)
    all_results = []
    print(f"\n[3/3] Walk-forward optimization  ({total_tasks} tasks, {n_workers} workers)...")

    if n_workers == 1:
        for i, t in enumerate(tasks, 1):
            strat_name, sym = t[0], t[2]
            print(f"  [{i}/{total_tasks}] {strat_name}@{sym}...", end=" ", flush=True)
            try:
                _, _, results = _scan_worker(t)
                if results:
                    all_results.extend(results)
                    best = max(results, key=lambda r: r["sharpe_ratio"])
                    print(f"Sharpe={best['sharpe_ratio']:.4f} ret={best['total_return_pct']:+.1f}%")
                else:
                    print("no valid results")
            except Exception as e:
                print(f"ERROR: {e}")
    else:
        with _PPE(max_workers=n_workers) as pool:
            future_map = {pool.submit(_scan_worker, t): (t[0], t[2]) for t in tasks}
            done = 0
            for fut in _asc(future_map):
                done += 1
                strat_name, sym = future_map[fut]
                try:
                    _, _, results = fut.result()
                    if results:
                        all_results.extend(results)
                        best = max(results, key=lambda r: r["sharpe_ratio"])
                        print(f"  [{done}/{total_tasks}] {strat_name}@{sym}: "
                              f"Sharpe={best['sharpe_ratio']:.4f}  "
                              f"ret={best['total_return_pct']:+.1f}%  "
                              f"overfit={best.get('overfit_pct', 'N/A')}%")
                    else:
                        print(f"  [{done}/{total_tasks}] {strat_name}@{sym}: no valid results")
                except Exception as e:
                    print(f"  [{done}/{total_tasks}] {strat_name}@{sym}: ERROR {e}")

    if not all_results:
        print("\nNo results. Strategies may not generate enough trades.")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("sharpe_ratio", ascending=False, na_position="last")

    print("\n" + "=" * 80)
    print("TOP 10 STRATEGY-SYMBOL COMBINATIONS (Validation Set)")
    print("=" * 80)

    param_cols = [c for c in results_df.columns if c.startswith("p_")]

    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        params_str = ", ".join(f"{c[2:]}={row[c]}" for c in param_cols if pd.notna(row.get(c)))
        print(f"\n  #{i}: {row['strategy']} @ {row['symbol']}")
        print(f"      Params: {params_str}")
        print(f"      Sharpe: {row['sharpe_ratio']:.4f}  |  "
              f"Return: {row['total_return_pct']:+.2f}%  |  "
              f"MaxDD: {row['max_drawdown_pct']:.2f}%")
        print(f"      Trades: {row['total_trades']}  |  "
              f"WinRate: {row['win_rate_pct']:.1f}%  |  "
              f"PF: {row['profit_factor']}  |  "
              f"Overfit: {row.get('overfit_pct', 'N/A')}%")
        print(f"      vs B&H: {row.get('vs_benchmark_pct', 'N/A'):+.2f}%  |  "
              f"Calmar: {row['calmar_ratio']:.4f}")

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")
    print(f"Total combos evaluated: {len(results_df)}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUTU-QUANT Parameter Scan")
    parser.add_argument("--output", default="results/scan_results.csv")
    parser.add_argument("--jobs", type=int, default=-1, help="并行进程数，-1=全部CPU")
    args = parser.parse_args()
    run_scan(args.output, n_jobs=args.jobs)
