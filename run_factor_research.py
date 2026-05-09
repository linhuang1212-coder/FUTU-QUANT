"""
FUTU-QUANT Factor Research — IC分析、分组回测、多因子评估

Usage:
    python run_factor_research.py --analyze                   # 全套因子IC分析
    python run_factor_research.py --score                     # 当前标的因子排名
    python run_factor_research.py --optimize-momentum         # 动量轮动最优参数
    python run_factor_research.py --analyze --symbols AAPL MSFT GOOGL  # 指定标的
"""
from __future__ import annotations

import sys
import io
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils.helpers import load_yaml, get_project_root
from utils.logger import setup_logger

logger = setup_logger("factor_research")

DEFAULT_SYMBOLS = [
    "US.AAPL", "US.MSFT", "US.GOOGL", "US.AMZN", "US.META",
    "US.NVDA", "US.TSLA", "US.SPY", "US.QQQ", "US.IWM",
]

SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech",
    "NVDA": "Tech", "AMD": "Tech", "AMZN": "Tech",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance",
    "JNJ": "Health", "PFE": "Health",
    "XOM": "Energy", "CVX": "Energy",
    "KO": "Consumer", "PG": "Consumer", "WMT": "Consumer", "TSLA": "Consumer",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF",
}

ETF_POOL = [
    "US.SGOV", "US.BIL", "US.TLT", "US.VEA",
    "US.EEM", "US.XLF", "US.XLE", "US.IWM",
]


def run_full_analysis(symbols: list[str], years: int = 5, lag: int = 5):
    """Run full factor IC analysis + quintile backtest."""
    from factor.data_provider import FactorDataProvider
    from factor.technical import build_all_technical
    from factor.volatility import calc_ivr, calc_hv_ratio
    from factor.processor import cross_sectional_rank, winsorize
    from factor.ic_analyzer import ic_report, print_ic_report
    from factor.quintile_backtest import quintile_backtest, print_quintile_report

    print("=" * 70)
    print(f"  Factor Research — {len(symbols)} symbols, {years}Y data, {lag}D forward")
    print("=" * 70)

    provider = FactorDataProvider()
    print(f"\n[1/5] Loading data for {len(symbols)} symbols...")
    prices, volumes = provider.get_daily_panel(symbols, years=years)
    if prices.empty:
        print("ERROR: No data loaded")
        return

    print(f"  Loaded: {len(prices)} days x {len(prices.columns)} symbols")
    print(f"  Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")

    print("\n[2/5] Building technical factors...")
    factors = build_all_technical(prices, volumes)

    print("  Building volatility factors...")
    factors["IVR"] = calc_ivr(prices)
    factors["HV_RATIO"] = calc_hv_ratio(prices)

    print(f"  Total factors: {len(factors)}")
    for name, panel in factors.items():
        valid_pct = panel.notna().sum().sum() / (panel.shape[0] * panel.shape[1]) * 100
        print(f"    {name:<12} valid={valid_pct:.1f}%")

    print("\n[3/5] Running IC analysis...")
    returns = prices.pct_change()
    ic_table, ic_series = ic_report(factors, returns, lag=lag)
    print()
    print_ic_report(ic_table, lag=lag)

    print("\n[4/5] Quintile backtest on top factors...")
    top_factors = ic_table.head(4).index.tolist()
    for fname in top_factors:
        if fname in factors:
            results = quintile_backtest(factors[fname], returns,
                                        n_groups=5, holding_period=lag)
            print()
            print_quintile_report(results, fname)

    print("\n[5/5] Generating report...")
    report = _generate_report(symbols, prices, factors, ic_table, ic_series,
                              returns, lag, top_factors)
    report_path = get_project_root() / "docs" / "factor_analysis_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to {report_path}")


def run_score(symbols: list[str]):
    """Print current factor scores for all symbols."""
    from factor.data_provider import FactorDataProvider
    from factor.technical import build_all_technical
    from factor.scorer import FactorScorer

    print("=" * 60)
    print("  Current Factor Scores")
    print("=" * 60)

    provider = FactorDataProvider()
    prices, volumes = provider.get_daily_panel(symbols, years=1)
    if prices.empty:
        print("ERROR: No data")
        return

    factors = build_all_technical(prices, volumes)
    scorer = FactorScorer(factors)
    ranking = scorer.rank_symbols()

    print(f"\n  Date: {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  {'Rank':<6}{'Symbol':<10}{'Score':<10}")
    print("  " + "-" * 26)
    for i, (sym, score) in enumerate(ranking, 1):
        print(f"  {i:<6}{sym:<10}{score:.4f}")


def run_optimize_momentum():
    """Evaluate different momentum lookback periods for ETF rotation."""
    from factor.data_provider import FactorDataProvider
    from factor.technical import calc_momentum
    from factor.ic_analyzer import ic_report, print_ic_report

    print("=" * 60)
    print("  Momentum Optimization for ETF Rotation")
    print("=" * 60)

    provider = FactorDataProvider()
    prices, _ = provider.get_daily_panel(ETF_POOL, years=5)
    if prices.empty:
        print("ERROR: No data for ETF pool")
        return

    print(f"  Loaded: {len(prices)} days x {len(prices.columns)} ETFs")

    windows = {"MOM_1M": 21, "MOM_2M": 42, "MOM_3M": 63, "MOM_6M": 126,
               "MOM_9M": 189, "MOM_12M": 252, "MOM_12M-1M": None}

    factors = {}
    for name, w in windows.items():
        if w is not None:
            factors[name] = calc_momentum(prices, w)
        else:
            factors[name] = calc_momentum(prices, 252) - calc_momentum(prices, 21)

    returns = prices.pct_change()

    for lag in [5, 21]:
        print(f"\n  Forward return period: {lag} days")
        ic_table, _ = ic_report(factors, returns, lag=lag)
        print()
        print_ic_report(ic_table, lag=lag)


def _generate_report(symbols, prices, factors, ic_table, ic_series,
                     returns, lag, top_factors) -> str:
    """Generate markdown analysis report."""
    lines = [
        f"# Factor Analysis Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## Data",
        f"- Symbols: {', '.join(s.split('.')[-1] for s in [c for c in prices.columns])}",
        f"- Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}",
        f"- Trading days: {len(prices)}",
        f"- Forward return: {lag} days",
        f"",
        f"## IC Analysis Summary",
        f"",
        f"| Factor | IC Mean | IC Std | IC_IR | t-stat | IC>0% |",
        f"|--------|---------|--------|-------|--------|-------|",
    ]
    for fname in ic_table.index:
        row = ic_table.loc[fname]
        lines.append(
            f"| {fname} | {row['IC_mean']:.4f} | {row['IC_std']:.4f} | "
            f"{row['IC_IR']:.4f} | {row['t_stat']:.2f} | "
            f"{row['IC_pos_ratio']:.1%} |"
        )
    lines.extend([
        f"",
        f"## Key Findings",
        f"",
        f"**Top factors by IC_IR:** {', '.join(top_factors)}",
        f"",
        f"### Interpretation",
        f"- IC_IR > 0.5: Strong, stable predictive power",
        f"- IC_IR 0.1-0.5: Moderate, usable in composite",
        f"- IC_IR < 0.1: Weak, limited standalone value",
        f"- |t-stat| > 2: Statistically significant",
        f"",
        f"## Recommendations for FUTU-QUANT",
        f"",
    ])

    strong = [f for f in ic_table.index if abs(ic_table.loc[f, "IC_IR"]) > 0.1]
    if strong:
        lines.append(f"Factors with IC_IR > 0.1: **{', '.join(strong)}**")
        lines.append(f"These are candidates for Credit Spread symbol ranking and "
                     f"momentum rotation optimization.")
    else:
        lines.append("No factors show strong IC_IR. Consider expanding the symbol pool "
                     "or testing different factor definitions.")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Factor Research")
    parser.add_argument("--analyze", action="store_true", help="Full IC analysis")
    parser.add_argument("--score", action="store_true", help="Current factor scores")
    parser.add_argument("--optimize-momentum", action="store_true",
                        help="Optimize momentum lookback for ETF rotation")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbol list")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--lag", type=int, default=5,
                        help="Forward return period in days")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS

    if args.analyze:
        run_full_analysis(symbols, years=args.years, lag=args.lag)
    elif args.score:
        run_score(symbols)
    elif args.optimize_momentum:
        run_optimize_momentum()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
