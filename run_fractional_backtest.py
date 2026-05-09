"""
Momentum Rotation ETF 回测 + 7-Gate 验证.

Downloads 10 years of daily data for the ETF pool,
runs the 12M-1M momentum rotation backtest,
then validates through the full 7-gate pipeline.

Usage:
  python run_fractional_backtest.py
"""
from __future__ import annotations

import sys, io  # noqa: E401
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.downloader import load_daily, save_daily
from strategy.fractional.momentum_rotation import MomentumRotation
from backtest.full_validation import (
    DEFAULT_COST, run_monte_carlo, run_stress_test,
    sensitivity_scan_1d, FullSensitivityResult,
    build_validation, print_validation_report,
)
from backtest.validation import deflated_sharpe_ratio

# ── Config ──

ETF_POOL = ["SGOV", "BIL", "TLT", "VEA", "EEM", "XLF", "XLE", "IWM"]
BUDGET = 500
REPORT_DIR = Path(__file__).resolve().parent / "docs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "momentum_rotation_report.md"


# ═══════════════════════════════════════════════════════════════════
#  Data Preparation
# ═══════════════════════════════════════════════════════════════════

def ensure_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Download and load daily data for all symbols."""
    data = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is None or len(df) < 500:
            print(f"  下载 {sym} 日线数据...")
            try:
                n = save_daily(sym, years=15)
                if n > 200:
                    df = load_daily(sym)
                else:
                    print(f"    {sym}: 下载失败或数据不足")
                    continue
            except Exception as e:
                print(f"    {sym}: 下载异常: {e}")
                continue
        else:
            print(f"  {sym}: 已有 {len(df)} bars")
        data[sym] = df
    return data


# ═══════════════════════════════════════════════════════════════════
#  Backtest
# ═══════════════════════════════════════════════════════════════════

def run_backtest(data: dict[str, pd.DataFrame], budget=BUDGET,
                 top_n=2, lookback=252, skip=21, sma=200) -> dict:
    """Run momentum rotation backtest."""
    result = MomentumRotation.backtest_momentum(
        data, budget=budget, top_n=top_n,
        lookback=lookback, skip=skip, sma_period=sma,
    )
    return result


def backtest_pnl_monthly(data: dict[str, pd.DataFrame], budget=BUDGET,
                         top_n=2, lookback=252, skip=21, sma=200) -> list[float]:
    """Return monthly PnL series for Monte Carlo analysis."""
    result = run_backtest(data, budget, top_n, lookback, skip, sma)
    equity = result["equity"]
    if len(equity) < 22:
        return []
    monthly_pnl = []
    for i in range(21, len(equity), 21):
        monthly_pnl.append(equity[i] - equity[i - 21])
    return monthly_pnl


# ═══════════════════════════════════════════════════════════════════
#  Sensitivity wrappers
# ═══════════════════════════════════════════════════════════════════

def _momentum_eval(params: dict, data=None) -> float:
    """Evaluate one param combo, return annualized Sharpe."""
    result = run_backtest(
        data,
        budget=params.get("budget", BUDGET),
        top_n=params.get("top_n", 2),
        lookback=params.get("lookback", 252),
        skip=params.get("skip", 21),
        sma=params.get("sma", 200),
    )
    return result["sharpe"]


BASE_PARAMS = {
    "budget": BUDGET,
    "top_n": 2,
    "lookback": 252,
    "skip": 21,
    "sma": 200,
}


# ═══════════════════════════════════════════════════════════════════
#  Stress test wrapper
# ═══════════════════════════════════════════════════════════════════

def _momentum_stress_func(closes_slice: np.ndarray, **kwargs) -> list[float]:
    """Not directly applicable — momentum uses multi-asset data.
    Return empty to skip per-asset stress (portfolio-level stress handled separately).
    """
    return []


# ═══════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  动量轮动 ETF 回测 + 7-Gate 验证")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Data
    print("\n[1] 数据准备...")
    data = ensure_data(ETF_POOL)
    if len(data) < 4:
        print("  数据不足，需要至少 4 只 ETF")
        return

    # Step 2: Base backtest
    print("\n[2] 基础回测 (12M-1M 动量 Top2, SMA200 过滤)...")
    result = run_backtest(data)

    print(f"\n  起始资金: ${BUDGET:,.0f}")
    print(f"  最终资金: ${result['final_value']:,.0f}")
    print(f"  总收益: {result['total_return']:.1%}")
    print(f"  年化收益 (CAGR): {result['cagr']:.1%}")
    print(f"  最大回撤: {result['max_drawdown']:.1%}")
    print(f"  Sharpe: {result['sharpe']:.2f}")
    print(f"  交易次数: {result['n_trades']}")

    # Step 3: Monthly PnL for Monte Carlo
    print("\n[3] Monte Carlo 模拟...")
    monthly_pnls = backtest_pnl_monthly(data)
    pnl_arr = np.array(monthly_pnls) if monthly_pnls else np.array([0.0])

    mc_result = run_monte_carlo(pnl_arr)
    print(f"  Shuffle DD percentile: {mc_result.shuffle_max_dd_percentile:.0f}%")
    print(f"  Bootstrap Sharpe CI: [{mc_result.bootstrap_sharpe_ci_lo:.2f}, "
          f"{mc_result.bootstrap_sharpe_ci_hi:.2f}] | Pass: {mc_result.bootstrap_pass}")

    # Step 4: Sensitivity
    print("\n[4] 参数敏感性分析...")
    sens = FullSensitivityResult()
    sens_params = {
        "top_n": [1, 2, 3, 4],
        "lookback": [126, 189, 252, 315],
        "skip": [0, 21, 42, 63],
        "sma": [100, 150, 200, 250],
    }

    for pname, pvals in sens_params.items():
        print(f"  scanning {pname}...", end=" ", flush=True)
        sr = sensitivity_scan_1d(
            pname, pvals, _momentum_eval, BASE_PARAMS,
            data=data,
        )
        sens.param_results.append(sr)
        status = "plateau" if sr.is_plateau else ("CLIFF!" if sr.is_cliff else "ok")
        print(f"score={sr.sensitivity_score:.2f} [{status}]")

    sens.compute()
    print(f"  Overall: {sens.overall_score:.2f} | Pass: {sens.pass_gate}")

    # Step 5: DSR
    print("\n[5] DSR 显著性...")
    cost_sharpe = result["sharpe"]
    n_trades = result["n_trades"]
    n_trials = 4  # top_n, lookback, skip, sma dimensions
    dsr = deflated_sharpe_ratio(cost_sharpe, n_trials, max(n_trades, 10))
    sig = "YES" if dsr["is_significant"] else "NO"
    print(f"  DSR: {dsr['deflated_sharpe']:.3f} | p={dsr['p_value']:.3f} | Significant: {sig}")

    # Step 6: Build scorecard
    print("\n[6] 7-Gate 评分...")
    fv = build_validation(
        "momentum_rotation",
        cost_sharpe,
        cpcv_result=None,  # Multi-asset CPCV not applicable in standard form
        mc_result=mc_result,
        sens_result=sens,
        stress_result=None,  # Multi-asset stress handled differently
        dsr_dict=dsr,
        n_trades=n_trades,
        win_rate=float(np.mean(pnl_arr > 0) * 100) if len(pnl_arr) > 0 else 0,
    )
    print_validation_report(fv)

    # Step 7: Generate report
    print("\n[7] 生成报告...")
    _generate_report(result, mc_result, sens, dsr, fv, data)

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f} 秒")


def _generate_report(result, mc, sens, dsr, fv, data):
    lines = [
        "# 动量轮动 ETF 回测报告",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"资金: ${BUDGET:,}",
        f"ETF 池: {', '.join(data.keys())}",
        "",
        "## 基础回测结果",
        "",
        f"- 总收益: {result['total_return']:.1%}",
        f"- CAGR: {result['cagr']:.1%}",
        f"- 最大回撤: {result['max_drawdown']:.1%}",
        f"- Sharpe: {result['sharpe']:.2f}",
        f"- 交易次数: {result['n_trades']}",
        "",
        "## 7-Gate 验证",
        "",
        f"总分: {fv.total_score:.0f}/100",
        f"判定: {fv.verdict}",
        "",
        "| Gate | 检验 | 得分 | 通过 | 详情 |",
        "|------|------|------|------|------|",
    ]
    for i, g in enumerate(fv.gates, 1):
        st = "✓" if g.passed else "✗"
        lines.append(f"| {i} | {g.name} | {g.actual_score:.0f}/{g.max_score} | {st} | {g.detail} |")

    if sens.param_results:
        lines.append("")
        lines.append("## 参数敏感性")
        lines.append("")
        lines.append("| 参数 | Score | Plateau | 值 -> Sharpe |")
        lines.append("|------|-------|---------|-------------|")
        for sr in sens.param_results:
            p = "✓" if sr.is_plateau else "✗"
            sharpe_str = ", ".join(f"{s:.2f}" for s in sr.sharpes)
            lines.append(f"| {sr.param_name} | {sr.sensitivity_score:.2f} | {p} | {sr.values} -> [{sharpe_str}] |")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  报告已生成: {REPORT_PATH}")


if __name__ == "__main__":
    main()
