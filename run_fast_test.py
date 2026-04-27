"""run_fast_test.py — 一键快速验证脚本（并行加速版）

功能：
  1. 快速验证当前最优策略（TQQQ+UGL 对冲组合 + Swing 策略）
  2. 多进程并行跑所有 symbol × strategy × segment 组合
  3. 输出结构化报告，不降低测试质量

用法：
    python run_fast_test.py                    # 完整验证（推荐）
    python run_fast_test.py --mode swing       # 只验证 Swing 策略
    python run_fast_test.py --mode hedge       # 只验证对冲组合
    python run_fast_test.py --mode scan        # 参数扫描（多进程）
    python run_fast_test.py --jobs 4           # 指定并行进程数
    python run_fast_test.py --quick            # 快速模式（只跑3yr数据）

Windows 注意：必须在 if __name__ == "__main__" 下运行，已处理。
"""

import sys
import io
import os
import time
import math
import argparse
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 路径修正（Windows 子进程兼容）───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.downloader import load_daily
from utils.logger import setup_logger

logger = setup_logger("fast_test")

# ── 配置 ─────────────────────────────────────────────────────────────────────

SWING_SYMBOLS = ["US.TQQQ", "US.SOXL"]
HEDGE_SYMBOLS = ["US.TQQQ", "US.UGL", "US.IEF", "US.QQQ"]

SEGMENTS = {
    "10yr": ("2016-01-01", "2026-04-30"),
    "5yr":  ("2021-01-01", "2026-04-30"),
    "3yr":  ("2023-01-01", "2026-04-30"),
}

STRESS_PERIODS = {
    "COVID":     ("2020-01-01", "2020-06-30"),
    "RateHike":  ("2022-01-01", "2022-12-31"),
    "AIBull":    ("2023-01-01", "2024-12-31"),
}

PASS_SHARPE = 0.8  # 实盘阶段合理目标（非论文级别）
PASS_MAX_DD = 65.0  # TQQQ 本身 MaxDD 就 80%+，组合 < 65% 可接受

STRATEGY_PARAMS = {
    "momentum": {
        "fast_ma_period": 8, "slow_ma_period": 15,
        "rsi_period": 10, "rsi_oversold": 30, "rsi_overbought": 70,
        "volume_ratio_threshold": 1.0, "cross_lookback": 3,
        "rsi_momentum_enabled": True, "ema_trend_enabled": True,
    },
    "breakout": {
        "lookback_period": 10, "volume_ratio_threshold": 1.2,
        "atr_breakout_multiplier": 1.5,
    },
    "mean_reversion": {
        "bb_period": 15, "bb_std": 2.0, "rsi_period": 14,
        "rsi_oversold": 25, "rsi_overbought": 75,
    },
    "rsi_reversal": {
        "rsi_period": 5, "rsi_buy_threshold": 25, "rsi_sell_threshold": 75,
    },
    "multi_factor": {
        "fast_ma_period": 10, "slow_ma_period": 15,
        "rsi_period": 10, "ema_period": 15,
        "buy_threshold": 3, "sell_threshold": 3,
    },
}

SYMBOL_STRATEGIES = {
    "US.TQQQ": ["momentum", "breakout", "mean_reversion", "multi_factor"],
    "US.SOXL": ["breakout", "mean_reversion", "rsi_reversal", "multi_factor"],
}


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def load_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for sym in symbols:
        ticker = sym.split(".")[-1] if "." in sym else sym
        df = load_daily(ticker)
        if df is not None and len(df) > 200:
            df["time_key"] = pd.to_datetime(df["time_key"])
            data[sym] = df
            print(f"  {sym}: {len(df)} bars  ({df['time_key'].iloc[0].date()} ~ {df['time_key'].iloc[-1].date()})")
        else:
            print(f"  {sym}: [NO DATA]")
    return data


def slice_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["time_key"] >= start) & (df["time_key"] <= end)
    return df.loc[mask].reset_index(drop=True)


def fast_backtest(signals, closes, commission=0.001, slippage=0.0005, capital=3000.0):
    pos = 0
    avg = 0.0
    cap = capital
    equity = np.empty(len(closes))
    trades = []

    for i in range(len(closes)):
        p = closes[i]
        s = signals[i]
        if s == 1 and pos == 0:
            buy_p = p * (1 + slippage)
            qty = int(cap * 0.95 / buy_p)
            if qty > 0:
                comm = buy_p * qty * commission
                cost = buy_p * qty + comm
                if cost <= cap:
                    pos = qty; avg = buy_p; cap -= cost
                    trades.append({"type": "BUY", "p": buy_p, "qty": qty})
        elif s == -1 and pos > 0:
            sell_p = p * (1 - slippage)
            comm = sell_p * pos * commission
            pnl = (sell_p - avg) * pos - comm
            cap += sell_p * pos - comm
            trades.append({"type": "SELL", "p": sell_p, "qty": pos, "pnl": pnl})
            pos = 0; avg = 0.0
        equity[i] = cap + pos * p

    final = cap + (pos * closes[-1] if pos > 0 else 0)
    return {"initial": capital, "final": final, "equity": equity, "trades": trades}


def compute_metrics(result: dict, closes: np.ndarray) -> dict:
    initial, final = result["initial"], result["final"]
    equity = result["equity"]
    sells = [t for t in result["trades"] if t["type"] == "SELL"]
    n = len(equity)

    ret_pct = (final / initial - 1) * 100
    years = max(n / 252, 0.01)
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if final > 0 else 0

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd)) * 100

    if len(sells) == 0:
        return {"sharpe": 0, "sortino": 0, "calmar": 0, "cagr": cagr,
                "max_dd": max_dd, "return_pct": ret_pct, "trades": 0,
                "win_rate": 0, "profit_factor": 0, "bh_return": (closes[-1]/closes[0]-1)*100}

    wins = [t["pnl"] for t in sells if t["pnl"] > 0]
    losses = [t["pnl"] for t in sells if t["pnl"] <= 0]
    win_rate = len(wins) / len(sells) * 100
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0)

    rets = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    rf = 1.05 ** (1 / 252) - 1
    excess = rets - rf
    std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0
    sharpe = float(np.mean(excess) / std * math.sqrt(252)) if std > 1e-12 else 0
    downside = np.minimum(0, excess)
    ddev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 0
    sortino = float(np.mean(excess) / ddev * math.sqrt(252)) if ddev > 1e-12 else 0
    calmar = (cagr / 100) / (max_dd / 100) if max_dd > 0.01 else 0
    bh = (closes[-1] / closes[0] - 1) * 100

    return {
        "sharpe": round(sharpe, 4), "sortino": round(sortino, 4),
        "calmar": round(calmar, 4), "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2), "return_pct": round(ret_pct, 2),
        "trades": len(sells), "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else pf,
        "bh_return": round(bh, 2),
    }


def precompute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for p in [5, 8, 10, 14, 15, 20, 50, 200]:
        df[f"ma_{p}"] = df["close"].rolling(p).mean()
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    for p in [5, 7, 10, 14]:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_g = gain.rolling(p).mean()
        avg_l = loss.rolling(p).mean()
        rs = avg_g / avg_l.replace(0, np.inf)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    for period, std in [(15, 2.0), (20, 2.0)]:
        mid = df["close"].rolling(period).mean()
        band = df["close"].rolling(period).std()
        df[f"bb_upper_{period}"] = mid + std * band
        df[f"bb_lower_{period}"] = mid - std * band
    # Volume ratio
    if "volume" in df.columns:
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma20"].clip(lower=1)
    else:
        df["vol_ratio"] = 1.0
    # ATR
    if "high" in df.columns and "low" in df.columns:
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        df["atr_14"] = pd.Series(tr).rolling(14).mean().values
    return df


# ── Swing 信号生成 ────────────────────────────────────────────────────────────

def gen_signals(df: pd.DataFrame, strat: str, params: dict):
    n = len(df)
    signals = np.zeros(n, dtype=int)

    if strat == "momentum":
        fast = df[f"ma_{params['fast_ma_period']}"].values
        slow = df[f"ma_{params['slow_ma_period']}"].values
        rsi = df[f"rsi_{params['rsi_period']}"].values
        ema_fast = df[f"ema_{params['fast_ma_period']}"].values
        ema_slow = df[f"ema_{params['slow_ma_period']}"].values
        in_pos = False
        for i in range(params["slow_ma_period"] + 5, n):
            if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(rsi[i]):
                continue
            crossed_up = fast[i] > slow[i] and fast[i-1] <= slow[i-1]
            crossed_dn = fast[i] < slow[i] and fast[i-1] >= slow[i-1]
            rsi_ok = rsi[i] > params["rsi_oversold"]
            trend_ok = ema_fast[i] > ema_slow[i] if params.get("ema_trend_enabled") else True
            if not in_pos and crossed_up and rsi_ok and trend_ok:
                signals[i] = 1; in_pos = True
            elif in_pos and (crossed_dn or rsi[i] > params["rsi_overbought"]):
                signals[i] = -1; in_pos = False

    elif strat == "breakout":
        lb = params["lookback_period"]
        close = df["close"].values
        atr = df.get("atr_14", pd.Series(np.full(n, np.nan))).values if "atr_14" in df.columns else np.full(n, np.nan)
        vol_ratio = df["vol_ratio"].values
        in_pos = False
        for i in range(lb + 5, n):
            if np.isnan(close[i]):
                continue
            prev_high = np.max(close[max(0, i-lb):i])
            prev_low = np.min(close[max(0, i-lb):i])
            atr_v = atr[i] if not np.isnan(atr[i]) else (prev_high - prev_low) * 0.1
            vol_ok = vol_ratio[i] >= params["volume_ratio_threshold"]
            if not in_pos and close[i] > prev_high + params["atr_breakout_multiplier"] * atr_v and vol_ok:
                signals[i] = 1; in_pos = True
            elif in_pos and close[i] < prev_low:
                signals[i] = -1; in_pos = False

    elif strat == "mean_reversion":
        bb_p = params["bb_period"]
        rsi = df[f"rsi_{params['rsi_period']}"].values
        upper = df[f"bb_upper_{bb_p}"].values if f"bb_upper_{bb_p}" in df.columns else np.full(n, np.nan)
        lower = df[f"bb_lower_{bb_p}"].values if f"bb_lower_{bb_p}" in df.columns else np.full(n, np.nan)
        close = df["close"].values
        in_pos = False
        for i in range(bb_p + 5, n):
            if np.isnan(lower[i]) or np.isnan(rsi[i]):
                continue
            if not in_pos and close[i] < lower[i] and rsi[i] < params["rsi_oversold"]:
                signals[i] = 1; in_pos = True
            elif in_pos and (close[i] > upper[i] or rsi[i] > params["rsi_overbought"]):
                signals[i] = -1; in_pos = False

    elif strat == "rsi_reversal":
        rsi = df[f"rsi_{params['rsi_period']}"].values
        in_pos = False
        for i in range(params["rsi_period"] + 5, n):
            if np.isnan(rsi[i]):
                continue
            if not in_pos and rsi[i] <= params["rsi_buy_threshold"]:
                signals[i] = 1; in_pos = True
            elif in_pos and rsi[i] >= params["rsi_sell_threshold"]:
                signals[i] = -1; in_pos = False

    elif strat == "multi_factor":
        fast = df[f"ma_{params['fast_ma_period']}"].values
        slow = df[f"ma_{params['slow_ma_period']}"].values
        rsi = df[f"rsi_{params['rsi_period']}"].values
        ema = df[f"ema_{params['ema_period']}"].values
        close = df["close"].values
        in_pos = False
        for i in range(params["slow_ma_period"] + 5, n):
            if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(rsi[i]):
                continue
            score = 0
            if fast[i] > slow[i]: score += 1
            if rsi[i] > 50: score += 1
            if close[i] > ema[i]: score += 1
            if not in_pos and score >= params["buy_threshold"]:
                signals[i] = 1; in_pos = True
            elif in_pos and score <= (3 - params["sell_threshold"]):
                signals[i] = -1; in_pos = False

    return signals


# ── 并行 Worker ───────────────────────────────────────────────────────────────

def _swing_worker(args):
    """单个 (symbol, strategy, segment_name) 验证任务。"""
    sym, strat, seg_name, seg_start, seg_end, full_df_records = args

    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    df = pd.DataFrame(full_df_records)
    df["time_key"] = pd.to_datetime(df["time_key"])
    seg_df = df.loc[(df["time_key"] >= seg_start) & (df["time_key"] <= seg_end)].reset_index(drop=True)

    if len(seg_df) < 50:
        return sym, strat, seg_name, None

    # 在子进程内再导入
    from run_fast_test import precompute, gen_signals, fast_backtest, compute_metrics, STRATEGY_PARAMS
    seg_df = precompute(seg_df)
    params = STRATEGY_PARAMS[strat]
    signals = gen_signals(seg_df, strat, params)
    closes = seg_df["close"].values
    result = fast_backtest(signals, closes)
    metrics = compute_metrics(result, closes)
    return sym, strat, seg_name, metrics


# ── 对冲组合验证（TQQQ + UGL 50/50 再平衡）────────────────────────────────────

def validate_hedge_portfolio(data: dict, quick: bool = False) -> dict:
    """验证 TQQQ+UGL 50/50 月度再平衡组合。"""
    if "US.TQQQ" not in data or "US.UGL" not in data:
        print("  [SKIP] 缺少 TQQQ 或 UGL 数据")
        return {}

    tqqq = data["US.TQQQ"].set_index("time_key")["close"].rename("TQQQ")
    ugl = data["US.UGL"].set_index("time_key")["close"].rename("UGL")
    qqq = data["US.QQQ"].set_index("time_key")["close"].rename("QQQ") if "US.QQQ" in data else None

    prices = pd.concat([tqqq, ugl], axis=1).dropna()
    if len(prices) < 200:
        print("  [SKIP] 对冲组合数据不足")
        return {}

    segs = {"3yr": ("2023-01-01", "2026-04-30")} if quick else SEGMENTS

    results = {}
    for seg_name, (start, end) in segs.items():
        seg = prices.loc[start:end]
        if len(seg) < 50:
            continue

        # 月度再平衡模拟
        capital = 3000.0
        tqqq_w, ugl_w = 0.5, 0.5
        tqqq_shares = capital * tqqq_w / seg["TQQQ"].iloc[0]
        ugl_shares = capital * ugl_w / seg["UGL"].iloc[0]

        # SMA200 + VIX 简化过滤（用 QQQ SMA200 代替）
        qqq_seg = qqq.loc[start:end] if qqq is not None else None

        equity_list = []
        rebal_day = 0
        crash_shelter = False

        for i, (date, row) in enumerate(seg.iterrows()):
            tqqq_val = tqqq_shares * row["TQQQ"]
            ugl_val = ugl_shares * row["UGL"]
            total = tqqq_val + ugl_val
            equity_list.append(total)

            # 崩盘过滤器：TQQQ 单日跌超 20%
            if i > 0:
                prev_tqqq = seg["TQQQ"].iloc[i-1]
                day_chg = (row["TQQQ"] - prev_tqqq) / prev_tqqq
                if day_chg < -0.20 and not crash_shelter:
                    # 转入避险（简化：全卖 TQQQ，持 UGL）
                    ugl_shares = total / row["UGL"]
                    tqqq_shares = 0
                    crash_shelter = True
                elif crash_shelter and day_chg > 0.05:
                    # 恢复
                    tqqq_shares = total * tqqq_w / row["TQQQ"]
                    ugl_shares = total * ugl_w / row["UGL"]
                    crash_shelter = False

            # 月度再平衡
            rebal_day += 1
            if rebal_day >= 21 and not crash_shelter:
                comm = total * 0.001  # 手续费估算
                tqqq_shares = (total - comm) * tqqq_w / row["TQQQ"]
                ugl_shares = (total - comm) * ugl_w / row["UGL"]
                rebal_day = 0

        equity = np.array(equity_list)
        final = equity[-1]
        ret_pct = (final / capital - 1) * 100
        years = max(len(equity) / 252, 0.01)
        cagr = ((final / capital) ** (1 / years) - 1) * 100

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd)) * 100

        rets = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        rf = 1.05 ** (1 / 252) - 1
        std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0
        sharpe = float(np.mean(rets - rf) / std * math.sqrt(252)) if std > 1e-12 else 0

        results[seg_name] = {
            "sharpe": round(sharpe, 4), "cagr": round(cagr, 2),
            "max_dd": round(max_dd, 2), "return_pct": round(ret_pct, 2),
        }

    return results


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_swing_validation(data: dict, quick: bool = False, n_jobs: int = -1) -> None:
    n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    segs = {"3yr": SEGMENTS["3yr"]} if quick else SEGMENTS

    # 构造任务
    tasks = []
    for sym, strats in SYMBOL_STRATEGIES.items():
        if sym not in data:
            continue
        df = data[sym]
        # 序列化为 records 传入子进程（避免 DataFrame 直接 pickle 的问题）
        df_records = df.to_dict("records")
        for strat in strats:
            for seg_name, (start, end) in segs.items():
                tasks.append((sym, strat, seg_name, start, end, df_records))

    total = len(tasks)
    print(f"  任务数: {total}  并行 workers: {n_workers}")

    # 收集结果
    collected: dict[tuple, dict] = {}  # (sym, strat, seg) -> metrics

    if n_workers == 1 or total <= 4:
        for t in tasks:
            sym, strat, seg_name, metrics = _swing_worker(t)
            if metrics:
                collected[(sym, strat, seg_name)] = metrics
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_swing_worker, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    sym, strat, seg_name, metrics = fut.result()
                    if metrics:
                        collected[(sym, strat, seg_name)] = metrics
                except Exception as e:
                    print(f"  [ERROR] {e}")

    # 打印报告
    print()
    print("=" * 90)
    print(f"  {'Symbol':<12} {'Strategy':<16} {'Seg':<8} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'Trades':>7} {'WR%':>6} {'PASS'}")
    print("  " + "-" * 88)

    all_pass = True
    for sym in SWING_SYMBOLS:
        if sym not in data:
            continue
        for strat in SYMBOL_STRATEGIES.get(sym, []):
            for seg_name in segs:
                m = collected.get((sym, strat, seg_name))
                if m is None:
                    continue
                passed = m["sharpe"] >= PASS_SHARPE and m["max_dd"] <= PASS_MAX_DD
                all_pass = all_pass and passed
                flag = "✓" if passed else "✗"
                print(
                    f"  {sym:<12} {strat:<16} {seg_name:<8} "
                    f"{m['sharpe']:>8.4f} {m.get('cagr', 0):>8.2f} "
                    f"{m['max_dd']:>8.2f} {m['trades']:>7} "
                    f"{m['win_rate']:>6.1f} {flag}"
                )

    print("=" * 90)
    print(f"  总体: {'✓ PASS' if all_pass else '✗ 部分策略未通过'}")


def main():
    parser = argparse.ArgumentParser(description="FUTU-QUANT 快速验证")
    parser.add_argument("--mode", choices=["swing", "hedge", "scan", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="只跑3yr数据（更快）")
    parser.add_argument("--jobs", type=int, default=-1, help="并行进程数，-1=全部CPU")
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "=" * 60)
    print(f"  FUTU-QUANT 快速验证  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"  模式: {args.mode.upper()}  |  快速: {args.quick}  |  进程数: {args.jobs if args.jobs > 0 else os.cpu_count()}")
    print("=" * 60)

    # 加载数据
    print("\n[数据加载]")
    all_syms = list(set(SWING_SYMBOLS + HEDGE_SYMBOLS))
    data = load_data(all_syms)

    if args.mode in ("hedge", "all"):
        print("\n[对冲组合验证] TQQQ + UGL 50/50")
        hedge_results = validate_hedge_portfolio(data, quick=args.quick)
        print(f"\n  {'Segment':<8} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'Return%':>9} {'PASS'}")
        print("  " + "-" * 50)
        for seg, m in hedge_results.items():
            passed = m["sharpe"] >= PASS_SHARPE and m["max_dd"] <= PASS_MAX_DD
            flag = "✓" if passed else "✗"
            print(f"  {seg:<8} {m['sharpe']:>8.4f} {m['cagr']:>8.2f} {m['max_dd']:>8.2f} {m['return_pct']:>9.2f} {flag}")

    if args.mode in ("swing", "all"):
        print("\n[Swing 策略验证]")
        run_swing_validation(data, quick=args.quick, n_jobs=args.jobs)

    if args.mode in ("scan", "all") and not args.quick:
        print("\n[参数扫描] 运行 run_param_scan.py...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "run_param_scan.py", "--output", "results/fast_test_scan.csv"],
                capture_output=False, text=True, cwd=str(ROOT)
            )
        except Exception as e:
            print(f"  [SKIP] 参数扫描失败: {e}")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
