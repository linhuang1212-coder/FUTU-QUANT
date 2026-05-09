"""
Turbo Validation — unified entry point for 7-Gate validation across all strategies.

Usage:
  python run_turbo_validation.py --all
  python run_turbo_validation.py -s credit_spread,wheel_csp
  python run_turbo_validation.py --list
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

from backtest.data_cache import DataCache
from backtest.strategy_adapters import get_all_tasks, get_task, ValidationTask
from backtest.full_validation import (
    FullValidationResult,
    FullSensitivityResult,
    print_validation_report,
    run_cpcv,
    run_monte_carlo,
    run_stress_test,
    sensitivity_scan_1d,
    build_validation,
)
from backtest.validation import deflated_sharpe_ratio

try:
    from backtest.parallel_runner import validate_strategy, validate_all
    _HAS_PARALLEL = True
except ImportError:
    _HAS_PARALLEL = False


ALL_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "IWM", "F", "SOFI", "AAL", "RIVN", "VALE", "PINS",
    "SGOV", "BIL", "TLT", "VEA", "EEM", "XLF", "XLE",
]


# ═══════════════════════════════════════════════════════════════════
#  Single-strategy validation (fallback when parallel_runner absent)
# ═══════════════════════════════════════════════════════════════════

def _sharpe(pnls):
    if len(pnls) < 3:
        return 0.0
    arr = np.array(pnls, dtype=float)
    s = np.std(arr)
    if s <= 0:
        return 0.0
    return float(np.mean(arr) / s * np.sqrt(12))


def _validate_task(task: ValidationTask, cache: DataCache) -> FullValidationResult:
    """Run 7-Gate validation on a single task, return FullValidationResult."""
    primary_sym = task.symbols[0] if task.symbols else "SPY"
    closes = cache.get_closes(primary_sym)

    if len(closes) < 300:
        print(f"  [SKIP] {task.name}: insufficient data ({len(closes)} bars)")
        fv = FullValidationResult(strategy_name=task.name)
        fv.verdict = "FAIL — 数据不足"
        return fv

    print(f"\n  ── {task.name} ({task.strategy_type}) ──")
    print(f"     主标的: {primary_sym} ({len(closes)} bars)")

    extra_kw = {}
    if task.strategy_type == "etf":
        daily_data = {}
        for sym in task.symbols:
            frame = cache.get_frame(sym)
            if not frame.empty:
                daily_data[sym] = frame
        extra_kw["daily_data"] = daily_data

    # Gate 2: cost-adjusted backtest
    print(f"     [Gate 2] 含成本回测...")
    try:
        pnls = task.backtest_func(closes, **task.default_params, **extra_kw)
    except Exception as exc:
        print(f"     [ERROR] backtest failed: {exc}")
        pnls = []

    pnl_arr = np.array(pnls, dtype=float) if pnls else np.array([0.0])
    n_trades = len(pnls)
    cost_sharpe = _sharpe(pnls)
    win_rate = float(np.mean(pnl_arr > 0) * 100) if len(pnl_arr) > 0 else 0.0
    print(f"     trades={n_trades} | Sharpe={cost_sharpe:.2f} | WR={win_rate:.1f}%")

    # Gate 3: CPCV
    cpcv_result = None
    if task.pnl_for_cpcv and n_trades >= 10:
        print(f"     [Gate 3] CPCV...")
        try:
            cpcv_result = run_cpcv(task.pnl_for_cpcv, closes,
                                   n_groups=10, k_test=2,
                                   **task.default_params, **extra_kw)
            print(f"     PBO={cpcv_result.pbo:.1%} | Pass={cpcv_result.pass_gate}")
        except Exception as exc:
            print(f"     [WARN] CPCV failed: {exc}")

    # Gate 4: Monte Carlo
    mc_result = None
    if n_trades >= 10:
        print(f"     [Gate 4] Monte Carlo...")
        try:
            noise_func = task.pnl_for_noise
            mc_kwargs = {**task.default_params, **extra_kw}
            mc_result = run_monte_carlo(
                pnl_arr,
                closes=closes if noise_func else None,
                strategy_func=noise_func,
                **mc_kwargs,
            )
            print(f"     Bootstrap CI=[{mc_result.bootstrap_sharpe_ci_lo:.2f},"
                  f" {mc_result.bootstrap_sharpe_ci_hi:.2f}]"
                  f" | Noise med={mc_result.noise_sharpe_median:.2f}")
        except Exception as exc:
            print(f"     [WARN] Monte Carlo failed: {exc}")

    # Gate 5: sensitivity
    sens = None
    if task.eval_for_sensitivity and task.param_grid:
        print(f"     [Gate 5] 参数敏感性...")
        try:
            sens = FullSensitivityResult()
            fixed_kw = {"closes": closes}
            if task.needs_ivr:
                fixed_kw["ivr_arr"] = cache.get_ivr(primary_sym)
            if task.strategy_type == "etf":
                fixed_kw["daily_data"] = extra_kw.get("daily_data")

            for pname, pvals in task.param_grid.items():
                sr = sensitivity_scan_1d(
                    pname, pvals, task.eval_for_sensitivity,
                    task.default_params, **fixed_kw,
                )
                sens.param_results.append(sr)
            sens.compute()
            print(f"     Overall score={sens.overall_score:.2f} | Pass={sens.pass_gate}")
        except Exception as exc:
            print(f"     [WARN] Sensitivity failed: {exc}")
            sens = None

    # Gate 6: stress test
    stress_result = None
    if task.stress_func:
        print(f"     [Gate 6] 压力测试...")
        try:
            df = cache.get_frame(primary_sym)
            if not df.empty and "time_key" in df.columns:
                stress_result = run_stress_test(
                    df, task.stress_func,
                    capital=task.capital, **task.default_params, **extra_kw,
                )
                veto_names = [p.name for p in stress_result.periods if p.veto]
                print(f"     Pass={stress_result.pass_gate}"
                      + (f" | Vetoed: {veto_names}" if veto_names else ""))
        except Exception as exc:
            print(f"     [WARN] Stress test failed: {exc}")

    # Gate 7: DSR
    dsr = None
    if n_trades >= 10:
        print(f"     [Gate 7] DSR...")
        try:
            n_trials = max(1, len(task.param_grid))
            dsr = deflated_sharpe_ratio(cost_sharpe, n_trials, n_trades)
            sig = "YES" if dsr["is_significant"] else "NO"
            print(f"     DSR={dsr['deflated_sharpe']:.3f} | p={dsr['p_value']:.3f} | Sig={sig}")
        except Exception as exc:
            print(f"     [WARN] DSR failed: {exc}")

    fv = build_validation(
        task.name, cost_sharpe, cpcv_result, mc_result,
        sens, stress_result, dsr, n_trades, win_rate,
    )
    return fv


# ═══════════════════════════════════════════════════════════════════
#  Portfolio synergy analysis
# ═══════════════════════════════════════════════════════════════════

def run_portfolio_synergy(results: list[FullValidationResult],
                          cache: DataCache) -> dict:
    """Combine PnL streams from passing strategies and compute portfolio metrics."""
    passing = [r for r in results if r.total_score >= 60]
    if len(passing) < 2:
        return {
            "combined_sharpe": 0.0,
            "combined_max_dd": 0.0,
            "n_strategies": len(passing),
            "correlation_matrix": {},
            "note": "需要至少 2 个 PASS/CONDITIONAL 策略进行组合分析",
        }

    pnl_series = {}
    for fv in passing:
        task = get_task(fv.strategy_name)
        if task is None:
            continue
        primary_sym = task.symbols[0] if task.symbols else "SPY"
        closes = cache.get_closes(primary_sym)
        if len(closes) < 300:
            continue

        extra_kw = {}
        if task.strategy_type == "etf":
            daily_data = {}
            for sym in task.symbols:
                frame = cache.get_frame(sym)
                if not frame.empty:
                    daily_data[sym] = frame
            extra_kw["daily_data"] = daily_data

        try:
            pnls = task.backtest_func(closes, **task.default_params, **extra_kw)
            if pnls:
                pnl_series[fv.strategy_name] = np.array(pnls, dtype=float)
        except Exception:
            continue

    if len(pnl_series) < 2:
        return {
            "combined_sharpe": 0.0,
            "combined_max_dd": 0.0,
            "n_strategies": len(pnl_series),
            "correlation_matrix": {},
            "note": "可用 PnL 序列不足",
        }

    max_len = max(len(v) for v in pnl_series.values())
    padded = {}
    for name, arr in pnl_series.items():
        if len(arr) < max_len:
            padded[name] = np.pad(arr, (0, max_len - len(arr)), constant_values=0)
        else:
            padded[name] = arr[:max_len]

    names = sorted(padded.keys())
    matrix = np.column_stack([padded[n] for n in names])

    combined = matrix.sum(axis=1)
    combined_sharpe = _sharpe(combined.tolist())

    cum = np.cumsum(combined)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    combined_max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    corr = {}
    if matrix.shape[1] >= 2:
        corr_matrix = np.corrcoef(matrix.T)
        for i, n1 in enumerate(names):
            corr[n1] = {}
            for j, n2 in enumerate(names):
                corr[n1][n2] = float(corr_matrix[i, j])

    return {
        "combined_sharpe": combined_sharpe,
        "combined_max_dd": combined_max_dd,
        "n_strategies": len(pnl_series),
        "strategy_names": names,
        "correlation_matrix": corr,
    }


# ═══════════════════════════════════════════════════════════════════
#  Report generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(results: list[FullValidationResult], synergy: dict,
                    elapsed: float, path: str):
    """Write a comprehensive Markdown validation report."""
    n_cores = os.cpu_count() or 1
    n_jobs = synergy.get("_n_jobs", n_cores)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Turbo Validation 报告")
    lines.append("")
    lines.append(f"生成时间: {ts}")
    lines.append(f"总耗时: {elapsed:.1f}s")
    lines.append(f"CPU 核心: {n_cores} (使用 {n_jobs} workers)")
    lines.append("")

    # ── Overview table ──
    lines.append("## 总览")
    lines.append("")
    lines.append("| 策略 | 类型 | 总分 | 判定 | Sharpe | PBO | MC CI | 敏感性 | 压力 | DSR |")
    lines.append("|------|------|------|------|--------|-----|-------|--------|------|-----|")

    for fv in results:
        task = get_task(fv.strategy_name)
        stype = task.strategy_type if task else "?"

        pbo_str = f"{fv.cpcv.pbo:.0%}" if fv.cpcv else "N/A"

        if fv.monte_carlo:
            mc_ci = (f"[{fv.monte_carlo.bootstrap_sharpe_ci_lo:.2f},"
                     f"{fv.monte_carlo.bootstrap_sharpe_ci_hi:.2f}]")
        else:
            mc_ci = "N/A"

        sens_str = f"{fv.sensitivity.overall_score:.2f}" if fv.sensitivity else "N/A"

        if fv.stress:
            stress_str = "PASS" if fv.stress.pass_gate else "FAIL"
        else:
            stress_str = "N/A"

        if fv.dsr_result:
            dsr_str = f"p={fv.dsr_result.get('p_value', 1):.2f}"
        else:
            dsr_str = "N/A"

        verdict_short = "PASS" if fv.total_score >= 75 else ("COND" if fv.total_score >= 60 else "FAIL")

        lines.append(
            f"| {fv.strategy_name} | {stype} | {fv.total_score:.0f}/100 | {verdict_short} "
            f"| {fv.cost_sharpe:.2f} | {pbo_str} | {mc_ci} | {sens_str} "
            f"| {stress_str} | {dsr_str} |"
        )

    # ── Deployment recommendations ──
    lines.append("")
    lines.append("## 推荐部署")
    lines.append("")

    passing = [fv for fv in results if fv.total_score >= 75]
    conditional = [fv for fv in results if 60 <= fv.total_score < 75]
    failing = [fv for fv in results if fv.total_score < 60]

    lines.append("### PASS (推荐部署)")
    if passing:
        for fv in passing:
            lines.append(f"- **{fv.strategy_name}** ({fv.total_score:.0f}/100) — {fv.verdict}")
    else:
        lines.append("- (无)")

    lines.append("")
    lines.append("### CONDITIONAL (需进一步验证)")
    if conditional:
        for fv in conditional:
            lines.append(f"- **{fv.strategy_name}** ({fv.total_score:.0f}/100) — {fv.verdict}")
    else:
        lines.append("- (无)")

    lines.append("")
    lines.append("### FAIL (不建议部署)")
    if failing:
        for fv in failing:
            lines.append(f"- **{fv.strategy_name}** ({fv.total_score:.0f}/100) — {fv.verdict}")
    else:
        lines.append("- (无)")

    # ── Detailed per-strategy results ──
    lines.append("")
    lines.append("## 各策略详细结果")

    for fv in results:
        lines.append("")
        lines.append(f"### {fv.strategy_name}")
        lines.append("")

        task = get_task(fv.strategy_name)
        thesis = task.economic_thesis if task else ""
        if thesis:
            lines.append(f"**经济学假设**: {thesis}")
            lines.append("")

        n_trades_str = "N/A"
        win_rate_str = "N/A"
        for g in fv.gates:
            if "Sharpe" in g.name:
                lines.append(f"**基准回测**: {g.detail}")
                break

        lines.append("")
        lines.append("| Gate | 分数 | 状态 | 详情 |")
        lines.append("|------|------|------|------|")
        for i, g in enumerate(fv.gates, 1):
            status = "PASS" if g.passed else "FAIL"
            detail_escaped = g.detail.replace("|", "\\|")
            lines.append(
                f"| {i}. {g.name} | {g.actual_score:.0f}/{g.max_score} "
                f"| {status} | {detail_escaped} |"
            )

        lines.append("")
        lines.append(f"**总分: {fv.total_score:.0f}/100 — {fv.verdict}**")

        # CPCV details
        if fv.cpcv:
            lines.append("")
            lines.append("#### CPCV 详情")
            lines.append(f"- 路径数: {fv.cpcv.n_paths}")
            lines.append(f"- PBO: {fv.cpcv.pbo:.1%}")
            lines.append(f"- 平均 OOS Sharpe: {fv.cpcv.avg_oos_sharpe:.2f}")
            lines.append(f"- 中位 OOS Sharpe: {fv.cpcv.median_oos_sharpe:.2f}")

        # Monte Carlo details
        if fv.monte_carlo:
            mc = fv.monte_carlo
            lines.append("")
            lines.append("#### Monte Carlo 详情")
            lines.append(f"- Shuffle DD percentile: {mc.shuffle_max_dd_percentile:.0f}%")
            lines.append(f"- Bootstrap Sharpe CI: [{mc.bootstrap_sharpe_ci_lo:.2f}, "
                         f"{mc.bootstrap_sharpe_ci_hi:.2f}]")
            lines.append(f"- Noise Sharpe median: {mc.noise_sharpe_median:.2f}")

        # Sensitivity details
        if fv.sensitivity and fv.sensitivity.param_results:
            lines.append("")
            lines.append("#### 参数敏感性详情")
            lines.append("| 参数 | Score | Plateau | Cliff |")
            lines.append("|------|-------|---------|-------|")
            for sr in fv.sensitivity.param_results:
                p = "✓" if sr.is_plateau else "✗"
                c = "⚠" if sr.is_cliff else "—"
                lines.append(f"| {sr.param_name} | {sr.sensitivity_score:.2f} | {p} | {c} |")

        # Stress details
        if fv.stress and fv.stress.periods:
            lines.append("")
            lines.append("#### 压力测试详情")
            lines.append("| 事件 | 交易数 | PnL | MaxDD | Veto |")
            lines.append("|------|--------|-----|-------|------|")
            for p in fv.stress.periods:
                veto = "YES" if p.veto else "—"
                lines.append(
                    f"| {p.name} | {p.n_trades} | ${p.total_pnl:+,.0f} "
                    f"| ${p.max_drawdown:+,.0f} | {veto} |"
                )

        lines.append("")
        lines.append("---")

    # ── Portfolio synergy ──
    lines.append("")
    lines.append("## 组合协同分析")
    lines.append("")

    if synergy.get("n_strategies", 0) >= 2:
        lines.append(f"组合 Sharpe: {synergy['combined_sharpe']:.2f}")
        lines.append(f"组合最大回撤: ${synergy['combined_max_dd']:+,.0f}")
        lines.append(f"参与策略数: {synergy['n_strategies']}")
        lines.append("")

        corr = synergy.get("correlation_matrix", {})
        strat_names = synergy.get("strategy_names", [])
        if corr and strat_names:
            lines.append("策略相关性矩阵:")
            lines.append("")
            header = "| |" + "|".join(f" {n} " for n in strat_names) + "|"
            sep = "|---|" + "|".join("---" for _ in strat_names) + "|"
            lines.append(header)
            lines.append(sep)
            for n1 in strat_names:
                row = f"| {n1} |"
                for n2 in strat_names:
                    val = corr.get(n1, {}).get(n2, 0)
                    row += f" {val:.2f} |"
                lines.append(row)
    else:
        note = synergy.get("note", "策略数不足，无法进行组合分析")
        lines.append(f"*{note}*")

    # ── Conclusion ──
    lines.append("")
    lines.append("## 结论与建议")
    lines.append("")

    if passing:
        lines.append(f"共 {len(passing)} 个策略通过完整验证，建议小仓位部署:")
        for fv in passing:
            lines.append(f"- **{fv.strategy_name}** (得分 {fv.total_score:.0f}/100)")
    if conditional:
        lines.append("")
        lines.append(f"共 {len(conditional)} 个策略为 CONDITIONAL，建议先进行 Paper Trading:")
        for fv in conditional:
            lines.append(f"- **{fv.strategy_name}** (得分 {fv.total_score:.0f}/100)")
    if failing:
        lines.append("")
        lines.append(f"共 {len(failing)} 个策略未通过验证，不建议部署:")
        for fv in failing:
            lines.append(f"- **{fv.strategy_name}** (得分 {fv.total_score:.0f}/100)")

    if synergy.get("n_strategies", 0) >= 2:
        lines.append("")
        lines.append(f"组合运行时预期 Sharpe 为 {synergy['combined_sharpe']:.2f}，"
                     f"最大回撤 ${synergy['combined_max_dd']:+,.0f}。")
        avg_corr_vals = []
        corr = synergy.get("correlation_matrix", {})
        strat_names = synergy.get("strategy_names", [])
        for i, n1 in enumerate(strat_names):
            for j, n2 in enumerate(strat_names):
                if i < j:
                    avg_corr_vals.append(corr.get(n1, {}).get(n2, 0))
        if avg_corr_vals:
            avg_corr = float(np.mean(avg_corr_vals))
            lines.append(f"策略间平均相关性: {avg_corr:.2f} "
                         f"({'低相关，组合效果好' if avg_corr < 0.3 else '中等相关' if avg_corr < 0.6 else '高相关，分散效果有限'})。")

    # Write file
    report_dir = Path(path).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines) + "\n"
    Path(path).write_text(report_text, encoding="utf-8")
    return report_text


# ═══════════════════════════════════════════════════════════════════
#  Console summary
# ═══════════════════════════════════════════════════════════════════

def print_summary(results: list[FullValidationResult], elapsed: float,
                  report_path: str):
    passing = [(fv.strategy_name, fv.total_score) for fv in results if fv.total_score >= 75]
    conditional = [(fv.strategy_name, fv.total_score) for fv in results if 60 <= fv.total_score < 75]
    failing = [(fv.strategy_name, fv.total_score) for fv in results if fv.total_score < 60]

    def _fmt(items):
        if not items:
            return "(无)"
        return ", ".join(f"{n} ({s:.0f}/100)" for n, s in items)

    print()
    print("=" * 60)
    print("  Turbo Validation 完成")
    print("=" * 60)
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  策略数: {len(results)}")
    print()
    print(f"  PASS:    {_fmt(passing)}")
    print(f"  COND:    {_fmt(conditional)}")
    print(f"  FAIL:    {_fmt(failing)}")
    print()
    print(f"  报告: {report_path}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Turbo Validation — 7-Gate 全策略验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", "-a", action="store_true",
                        help="运行所有策略的验证")
    parser.add_argument("--strategy", "-s", type=str, default="",
                        help="运行指定策略 (逗号分隔, e.g. credit_spread,wheel_csp)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出所有可用策略并退出")
    parser.add_argument("--jobs", "-j", type=int, default=0,
                        help="并行 workers 数 (0=auto)")
    parser.add_argument("--report", type=str,
                        default="docs/turbo_validation_report.md",
                        help="输出报告路径")
    parser.add_argument("--no-download", action="store_true",
                        help="跳过下载缺失数据")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    all_tasks = get_all_tasks()
    task_map = {t.name: t for t in all_tasks}

    # --list: show available strategies and exit
    if args.list:
        print("\n可用策略:")
        print("-" * 50)
        for t in all_tasks:
            ivr_tag = " [IVR]" if t.needs_ivr else ""
            print(f"  {t.name:<25} {t.strategy_type:<10} symbols={t.symbols}{ivr_tag}")
        print(f"\n共 {len(all_tasks)} 个策略")
        return

    # Determine which tasks to run
    if args.all:
        tasks = list(all_tasks)
    elif args.strategy:
        names = [n.strip() for n in args.strategy.split(",") if n.strip()]
        tasks = []
        for n in names:
            t = task_map.get(n)
            if t is None:
                print(f"[WARN] 策略 '{n}' 不存在，已跳过。可用: {list(task_map.keys())}")
            else:
                tasks.append(t)
    else:
        print("请指定 --all 或 --strategy NAME 。使用 --list 查看可用策略。")
        parse_args(["--help"])
        return

    if not tasks:
        print("[ERROR] 无策略可运行")
        return

    n_jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    report_path = args.report
    download_missing = not args.no_download

    t0 = time.time()

    print("=" * 60)
    print("  Turbo Validation — 7-Gate 全策略验证")
    print("=" * 60)
    print(f"  策略: {[t.name for t in tasks]}")
    print(f"  Workers: {n_jobs}")
    print(f"  报告: {report_path}")
    print()

    # ── Step 1: collect symbols & load data ──
    all_syms = set()
    for t in tasks:
        all_syms.update(t.symbols)
    all_syms.update(ALL_SYMBOLS)
    all_syms = sorted(all_syms)

    print(f"  加载数据: {len(all_syms)} 个标的...")
    cache = DataCache.get()
    cache.load_symbols(all_syms, download_missing=download_missing)
    print(f"  {cache.summary()}")

    # Pre-warm IVR cache for options strategies
    ivr_symbols = set()
    for t in tasks:
        if t.needs_ivr:
            ivr_symbols.update(t.symbols)
    for sym in ivr_symbols:
        print(f"  预计算 IVR: {sym}...", end=" ", flush=True)
        ivr = cache.get_ivr(sym)
        print(f"({len(ivr)} bars)")

    # ── Step 2: run validation ──
    print(f"\n{'─'*60}")
    print("  开始验证")
    print(f"{'─'*60}")

    results: list[FullValidationResult] = []

    if _HAS_PARALLEL and len(tasks) > 1 and n_jobs > 1:
        try:
            all_results = validate_all(tasks, cache, n_jobs=n_jobs)
            results.extend(all_results)
        except Exception as exc:
            print(f"  [WARN] parallel_runner failed ({exc}), falling back to sequential")
            _HAS_PARALLEL_FALLBACK = True
            for task in tasks:
                try:
                    fv = _validate_task(task, cache)
                    results.append(fv)
                    print_validation_report(fv)
                except Exception as task_exc:
                    print(f"  [ERROR] {task.name} failed: {task_exc}")
                    fv = FullValidationResult(strategy_name=task.name)
                    fv.verdict = f"FAIL — 运行错误: {task_exc}"
                    results.append(fv)
    else:
        for task in tasks:
            try:
                fv = _validate_task(task, cache)
                results.append(fv)
                print_validation_report(fv)
            except Exception as task_exc:
                print(f"  [ERROR] {task.name} failed: {task_exc}")
                fv = FullValidationResult(strategy_name=task.name)
                fv.verdict = f"FAIL — 运行错误: {task_exc}"
                results.append(fv)

    # ── Step 3: portfolio synergy ──
    print(f"\n{'─'*60}")
    print("  组合协同分析")
    print(f"{'─'*60}")

    synergy = run_portfolio_synergy(results, cache)
    synergy["_n_jobs"] = n_jobs

    if synergy.get("n_strategies", 0) >= 2:
        print(f"  组合 Sharpe: {synergy['combined_sharpe']:.2f}")
        print(f"  组合最大回撤: ${synergy['combined_max_dd']:+,.0f}")
        print(f"  参与策略: {synergy.get('strategy_names', [])}")
    else:
        print(f"  {synergy.get('note', '策略数不足')}")

    # ── Step 4: generate report ──
    elapsed = time.time() - t0
    generate_report(results, synergy, elapsed, report_path)
    print(f"\n  报告已写入: {report_path}")

    # ── Step 5: console summary ──
    print_summary(results, elapsed, report_path)


if __name__ == "__main__":
    main()
