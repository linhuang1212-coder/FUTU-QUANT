"""
Full Quantitative Validation Pipeline — Entry Script.

Runs the complete 7-gate validation for Credit Spread strategy:
  Gate 1: Economic thesis
  Gate 2: Cost-adjusted backtest
  Gate 3: CPCV (Combinatorial Purged Cross-Validation)
  Gate 4: Monte Carlo simulation
  Gate 5: Parameter sensitivity
  Gate 6: Stress testing
  Gate 7: DSR significance

Usage:
  python run_full_validation.py
"""
from __future__ import annotations

import sys, io  # noqa: E401
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from data.downloader import load_daily
from options.pricer import compute_ivr
from options.backtest import _synth_iv

from backtest.full_validation import (
    CostModel, DEFAULT_COST,
    sim_spread_with_cost,
    cpcv_splits, run_cpcv, CPCVResult,
    run_monte_carlo, MonteCarloResult,
    sensitivity_scan_1d, FullSensitivityResult,
    run_stress_test, StressTestResult,
    build_validation, print_validation_report, FullValidationResult,
)
from backtest.validation import deflated_sharpe_ratio

# ── Config ──

CREDIT_SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

REPORT_DIR = Path(__file__).resolve().parent / "docs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "full_validation_report.md"

CAPITAL = 3000


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _precompute_ivr(closes: np.ndarray) -> np.ndarray:
    n = len(closes)
    rets = np.diff(np.log(closes))
    ivr_arr = np.full(n, np.nan)
    for i in range(252, n):
        current_vol = float(np.std(rets[i - 20:i]) * np.sqrt(252))
        vols = []
        for j in range(max(20, i - 252), i, 5):
            wv = float(np.std(rets[max(0, j - 20):j]) * np.sqrt(252))
            vols.append(wv)
        if vols:
            ivr_arr[i] = compute_ivr(current_vol, vols)
    return ivr_arr


def _compute_vol(closes: np.ndarray, idx: int, window: int = 20) -> float:
    rets = np.diff(np.log(closes[max(0, idx - window):idx + 1]))
    if len(rets) < 5:
        return 0.3
    return float(np.std(rets) * np.sqrt(252))


# ═══════════════════════════════════════════════════════════════════
#  Credit Spread: strategy function wrappers
# ═══════════════════════════════════════════════════════════════════

CS_BASE_PARAMS = {
    "spread_width": 5.0,
    "target_delta": 0.30,
    "max_hold": 21,
    "tp_pct": 0.50,
    "sl_pct": 2.00,
    "min_ivr": 60.0,
}


def _run_cs_backtest(closes: np.ndarray, ivr_arr: np.ndarray = None,
                     spread_width=5.0, target_delta=0.30, max_hold=21,
                     tp_pct=0.50, sl_pct=2.00, min_ivr=60.0,
                     cost_model=DEFAULT_COST, r=0.05) -> list[float]:
    """Run credit spread backtest on daily close array, return PnL list."""
    if ivr_arr is None:
        ivr_arr = _precompute_ivr(closes)
    rets = np.diff(np.log(closes))
    pnls = []
    i = 252
    while i < len(closes) - max_hold:
        ivr = ivr_arr[i] if i < len(ivr_arr) else np.nan
        if np.isnan(ivr) or ivr < min_ivr:
            i += 1
            continue
        spot = closes[i]
        if spot <= 0:
            i += 1
            continue
        vol = _compute_vol(closes, i)
        iv = _synth_iv(vol, max_hold)
        T = max_hold / 252

        result = sim_spread_with_cost(
            spot, iv, T, spread_width, target_delta, max_hold,
            tp_pct, sl_pct, r, closes, i, direction="BULL",
            cost_model=cost_model,
        )
        if result["credit"] > 0:
            pnls.append(result["pnl"])
        i += max_hold + 1
    return pnls


def _cs_pnl_for_cpcv(data: np.ndarray, start_idx: int, end_idx: int,
                      **kwargs) -> list[float]:
    """CPCV-compatible wrapper: returns PnLs for a data slice."""
    slice_closes = data[max(0, start_idx - 252):end_idx]
    ivr_arr = _precompute_ivr(slice_closes)
    return _run_cs_backtest(slice_closes, ivr_arr, **kwargs)


def _cs_pnl_for_noise(closes: np.ndarray, **kwargs) -> list[float]:
    """Noise injection wrapper."""
    ivr_arr = _precompute_ivr(closes)
    return _run_cs_backtest(closes, ivr_arr, **kwargs)


def _cs_eval_for_sensitivity(params: dict, closes=None, ivr_arr=None) -> float:
    """Sensitivity scan: evaluate one param combo, return Sharpe."""
    pnls = _run_cs_backtest(
        closes, ivr_arr,
        spread_width=params.get("spread_width", 5.0),
        target_delta=params.get("target_delta", 0.30),
        max_hold=params.get("max_hold", 21),
        tp_pct=params.get("tp_pct", 0.50),
        sl_pct=params.get("sl_pct", 2.00),
        min_ivr=params.get("min_ivr", 60.0),
    )
    if len(pnls) < 5:
        return -999.0
    arr = np.array(pnls)
    return float(np.mean(arr) / np.std(arr) * np.sqrt(12)) if np.std(arr) > 0 else 0.0


def _cs_stress_func(closes_slice: np.ndarray, **kwargs) -> list[float]:
    """Stress test wrapper."""
    if len(closes_slice) < 260:
        return []
    ivr_arr = _precompute_ivr(closes_slice)
    return _run_cs_backtest(closes_slice, ivr_arr, **kwargs)


# ═══════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def validate_credit_spread() -> FullValidationResult:
    print("\n" + "=" * 70)
    print("  Credit Spread 完整验证")
    print("=" * 70)

    # Load merged data
    all_closes = []
    all_dfs = []
    for sym in CREDIT_SYMBOLS:
        df = load_daily(sym)
        if df is not None and len(df) > 300:
            all_dfs.append(df)
    if not all_dfs:
        print("  [ERROR] 无可用数据")
        return FullValidationResult(strategy_name="credit_spread")

    primary_df = max(all_dfs, key=len)
    closes = primary_df["close"].values

    print(f"  主标的: {len(closes)} bars")
    ivr_arr = _precompute_ivr(closes)

    # ── Gate 2: Cost-adjusted backtest ──
    print("\n  [Gate 2] 含成本回测...")
    pnls = _run_cs_backtest(closes, ivr_arr)
    n_trades = len(pnls)
    pnl_arr = np.array(pnls) if pnls else np.array([0.0])
    cost_sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(12)) if np.std(pnl_arr) > 0 else 0
    win_rate = float(np.mean(pnl_arr > 0) * 100) if len(pnl_arr) > 0 else 0
    total_pnl = float(np.sum(pnl_arr))
    print(f"    交易: {n_trades} | 总盈亏: ${total_pnl:+,.0f} | "
          f"胜率: {win_rate:.1f}% | Sharpe: {cost_sharpe:.2f}")

    # ── Gate 3: CPCV ──
    print("\n  [Gate 3] CPCV (45 paths)...")
    cpcv_result = run_cpcv(_cs_pnl_for_cpcv, closes, n_groups=10, k_test=2)
    print(f"    PBO: {cpcv_result.pbo:.1%} | Avg OOS Sharpe: {cpcv_result.avg_oos_sharpe:.2f} | "
          f"Median: {cpcv_result.median_oos_sharpe:.2f} | Pass: {cpcv_result.pass_gate}")

    # ── Gate 4: Monte Carlo ──
    print("\n  [Gate 4] Monte Carlo...")
    mc_result = run_monte_carlo(
        pnl_arr, closes=closes, strategy_func=_cs_pnl_for_noise,
    )
    print(f"    Shuffle DD percentile: {mc_result.shuffle_max_dd_percentile:.0f}%")
    print(f"    Bootstrap Sharpe CI: [{mc_result.bootstrap_sharpe_ci_lo:.2f}, "
          f"{mc_result.bootstrap_sharpe_ci_hi:.2f}] | Pass: {mc_result.bootstrap_pass}")
    print(f"    Noise Sharpe median: {mc_result.noise_sharpe_median:.2f} | Pass: {mc_result.noise_pass}")

    # ── Gate 5: Sensitivity ──
    print("\n  [Gate 5] 参数敏感性分析...")
    sens = FullSensitivityResult()

    cs_sens_params = {
        "spread_width": [1.0, 2.5, 5.0, 7.5, 10.0],
        "target_delta": [0.15, 0.20, 0.25, 0.30, 0.35],
        "tp_pct": [0.30, 0.50, 0.75],
        "sl_pct": [1.0, 1.5, 2.0, 3.0],
        "min_ivr": [40, 50, 60, 70],
    }

    for pname, pvals in cs_sens_params.items():
        print(f"    scanning {pname}...", end=" ", flush=True)
        sr = sensitivity_scan_1d(
            pname, pvals, _cs_eval_for_sensitivity, CS_BASE_PARAMS,
            closes=closes, ivr_arr=ivr_arr,
        )
        sens.param_results.append(sr)
        status = "plateau" if sr.is_plateau else ("CLIFF!" if sr.is_cliff else "ok")
        print(f"score={sr.sensitivity_score:.2f} [{status}]")

    sens.compute()
    print(f"    Overall sensitivity_score: {sens.overall_score:.2f} | Pass: {sens.pass_gate}")

    # ── Gate 6: Stress test ──
    print("\n  [Gate 6] 压力测试...")
    stress_result = run_stress_test(primary_df, _cs_stress_func, capital=CAPITAL)
    for p in stress_result.periods:
        veto_str = " *** VETO ***" if p.veto else ""
        print(f"    {p.name}: {p.n_trades}笔 | PnL=${p.total_pnl:+,.0f} | "
              f"MaxDD=${p.max_drawdown:+,.0f} | Loss%={p.max_loss_pct:.1%}{veto_str}")
    print(f"    Pass: {stress_result.pass_gate}")

    # ── Gate 7: DSR ──
    print("\n  [Gate 7] DSR 显著性...")
    n_trials = 5  # spread_width * delta * tp * sl * ivr dimensions
    dsr = deflated_sharpe_ratio(cost_sharpe, n_trials, n_trades)
    sig = "YES" if dsr["is_significant"] else "NO"
    print(f"    DSR: {dsr['deflated_sharpe']:.3f} | p={dsr['p_value']:.3f} | Significant: {sig}")

    # ── Build scorecard ──
    fv = build_validation(
        "credit_spread", cost_sharpe, cpcv_result, mc_result,
        sens, stress_result, dsr, n_trades, win_rate,
    )
    print_validation_report(fv)
    return fv


# ═══════════════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════════════

def generate_full_report(results: list[FullValidationResult]):
    lines = [
        "# 完整量化验证管线报告",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"资本: ${CAPITAL:,}",
        "",
    ]

    lines.append("## 总览")
    lines.append("")
    lines.append("| 策略 | 总分 | 判定 |")
    lines.append("|------|------|------|")
    for fv in results:
        lines.append(f"| {fv.strategy_name} | {fv.total_score:.0f}/100 | {fv.verdict} |")

    for fv in results:
        lines.append("")
        lines.append(f"## {fv.strategy_name}")
        lines.append("")
        lines.append("| Gate | 检验项目 | 得分 | 通过 | 详情 |")
        lines.append("|------|---------|------|------|------|")
        for i, g in enumerate(fv.gates, 1):
            status = "✓" if g.passed else "✗"
            lines.append(f"| {i} | {g.name} | {g.actual_score:.0f}/{g.max_score} | {status} | {g.detail} |")

        # CPCV details
        if fv.cpcv:
            lines.append("")
            lines.append(f"### CPCV 详情")
            lines.append(f"- 路径数: {fv.cpcv.n_paths}")
            lines.append(f"- PBO: {fv.cpcv.pbo:.1%}")
            lines.append(f"- 平均 OOS Sharpe: {fv.cpcv.avg_oos_sharpe:.2f}")
            lines.append(f"- 中位 OOS Sharpe: {fv.cpcv.median_oos_sharpe:.2f}")

        # Monte Carlo details
        if fv.monte_carlo:
            mc = fv.monte_carlo
            lines.append("")
            lines.append(f"### Monte Carlo 详情")
            lines.append(f"- Shuffle DD percentile: {mc.shuffle_max_dd_percentile:.0f}%")
            lines.append(f"- Bootstrap Sharpe CI: [{mc.bootstrap_sharpe_ci_lo:.2f}, "
                        f"{mc.bootstrap_sharpe_ci_hi:.2f}]")
            lines.append(f"- Noise Sharpe median: {mc.noise_sharpe_median:.2f}")

        # Sensitivity details
        if fv.sensitivity and fv.sensitivity.param_results:
            lines.append("")
            lines.append(f"### 参数敏感性详情")
            lines.append("| 参数 | Score | Plateau | Cliff |")
            lines.append("|------|-------|---------|-------|")
            for sr in fv.sensitivity.param_results:
                p = "✓" if sr.is_plateau else "✗"
                c = "⚠" if sr.is_cliff else "—"
                lines.append(f"| {sr.param_name} | {sr.sensitivity_score:.2f} | {p} | {c} |")
                lines.append(f"| | values: {sr.values} | | |")
                lines.append(f"| | sharpes: {[f'{s:.2f}' for s in sr.sharpes]} | | |")

        # Stress details
        if fv.stress:
            lines.append("")
            lines.append(f"### 压力测试详情")
            lines.append("| 事件 | 交易数 | PnL | MaxDD | Veto |")
            lines.append("|------|--------|-----|-------|------|")
            for p in fv.stress.periods:
                veto = "YES" if p.veto else "—"
                lines.append(f"| {p.name} | {p.n_trades} | ${p.total_pnl:+,.0f} | "
                            f"${p.max_drawdown:+,.0f} | {veto} |")

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\n完整验证报告已生成: {REPORT_PATH}")
    return report_text


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  完整量化验证管线 (7-Gate Validation) — Credit Spread")
    print("=" * 70)
    t0 = time.time()

    fv_cs = validate_credit_spread()
    generate_full_report([fv_cs])

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
