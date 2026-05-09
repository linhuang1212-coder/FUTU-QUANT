"""
因子回测引擎 — 大规模 IC/IR 分析 + 多因子模型验证

基于 factor_library 的 5,800+ 股票 x 5 年历史数据:
  1. 单因子 IC/IR 分析 (Rank IC, Spearman)
  2. 五分位组合回测 (Long-Short)
  3. 多因子模型回测 (预设模型的历史累计收益)
  4. IC 衰减分析 (不同持仓周期)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils.logger import setup_logger

logger = setup_logger("factor_library.backtest")


def calc_ic_from_library(factor_df: pd.DataFrame,
                         prices: pd.DataFrame,
                         factor_name: str,
                         holding_days: int = 5,
                         method: str = "spearman") -> pd.Series:
    """Calculate factor IC time series from factor library data.

    Args:
        factor_df: MultiIndex (date, symbol) factor DataFrame
        prices: long-format prices (date, symbol, close)
        factor_name: column name in factor_df
        holding_days: forward return horizon
        method: "spearman" or "pearson"

    Returns:
        pd.Series: IC time series indexed by date
    """
    if factor_name not in factor_df.columns:
        return pd.Series(dtype=float)

    close_pivot = prices.pivot_table(index="date", columns="symbol", values="close")
    fwd_ret = close_pivot.pct_change(holding_days).shift(-holding_days)

    if not isinstance(factor_df.index, pd.MultiIndex):
        return pd.Series(dtype=float)

    dates = factor_df.index.get_level_values(0).unique().sort_values()
    ic_values = []
    ic_dates = []

    for date in dates:
        try:
            factor_cross = factor_df.loc[date, factor_name]
        except KeyError:
            continue

        if date not in fwd_ret.index:
            continue

        ret_cross = fwd_ret.loc[date]
        common = factor_cross.dropna().index.intersection(ret_cross.dropna().index)

        if len(common) < 30:
            continue

        f_vals = factor_cross[common].values
        r_vals = ret_cross[common].values

        if method == "spearman":
            corr, _ = sp_stats.spearmanr(f_vals, r_vals)
        else:
            corr, _ = sp_stats.pearsonr(f_vals, r_vals)

        if not np.isnan(corr):
            ic_values.append(corr)
            ic_dates.append(date)

    return pd.Series(ic_values, index=ic_dates, name=factor_name)


def ic_summary(ic: pd.Series) -> dict:
    """Comprehensive IC statistics."""
    ic_clean = ic.dropna()
    if len(ic_clean) < 5:
        return {"IC_mean": np.nan, "IC_std": np.nan, "IC_IR": np.nan,
                "IC_positive_pct": np.nan, "t_stat": np.nan, "n_periods": 0}

    ic_mean = ic_clean.mean()
    ic_std = ic_clean.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    t_stat = ic_mean / (ic_std / np.sqrt(len(ic_clean))) if ic_std > 0 else 0.0

    return {
        "IC_mean": round(ic_mean, 4),
        "IC_std": round(ic_std, 4),
        "IC_IR": round(ic_ir, 4),
        "IC_positive_pct": round((ic_clean > 0).mean(), 4),
        "t_stat": round(t_stat, 2),
        "n_periods": len(ic_clean),
    }


def run_ic_analysis(factor_dfs: dict[str, pd.DataFrame],
                    prices: pd.DataFrame,
                    holding_days: int = 5,
                    top_n: int = 20) -> pd.DataFrame:
    """Batch IC analysis across all factors from all categories.

    Returns sorted summary table (by IC_IR).
    """
    all_results = {}

    for category, df in factor_dfs.items():
        if df.empty or not isinstance(df.index, pd.MultiIndex):
            continue
        for factor_name in df.columns:
            logger.info(f"  IC: {category}/{factor_name}...")
            ic = calc_ic_from_library(df, prices, factor_name,
                                       holding_days=holding_days)
            if len(ic) > 0:
                summary = ic_summary(ic)
                summary["category"] = category
                all_results[factor_name] = summary

    if not all_results:
        return pd.DataFrame()

    table = pd.DataFrame(all_results).T
    table.index.name = "Factor"
    table = table.sort_values("IC_IR", ascending=False)
    return table.head(top_n) if top_n else table


def ic_decay_analysis(factor_df: pd.DataFrame,
                      prices: pd.DataFrame,
                      factor_name: str,
                      horizons: list[int] = None) -> pd.DataFrame:
    """IC decay: how IC changes with different holding periods."""
    if horizons is None:
        horizons = [1, 3, 5, 10, 21, 42, 63]

    results = []
    for h in horizons:
        ic = calc_ic_from_library(factor_df, prices, factor_name,
                                   holding_days=h)
        summary = ic_summary(ic)
        summary["holding_days"] = h
        results.append(summary)

    return pd.DataFrame(results).set_index("holding_days")


def quintile_backtest(factor_df: pd.DataFrame,
                      prices: pd.DataFrame,
                      factor_name: str,
                      n_groups: int = 5,
                      holding_days: int = 21,
                      rebalance_freq: int = 21) -> dict:
    """Factor quintile portfolio backtest.

    Sorts stocks into N groups by factor value, computes equal-weight returns.
    """
    if factor_name not in factor_df.columns:
        return {"group_stats": pd.DataFrame(), "monotonicity": 0.0}

    close_pivot = prices.pivot_table(index="date", columns="symbol", values="close")
    returns = close_pivot.pct_change()

    dates = factor_df.index.get_level_values(0).unique().sort_values()
    rebalance_dates = dates[::rebalance_freq]

    group_returns: dict[str, list[float]] = {
        f"G{i}": [] for i in range(1, n_groups + 1)}
    group_dates: list = []

    for i, date in enumerate(rebalance_dates):
        try:
            factor_cross = factor_df.loc[date, factor_name].dropna()
        except KeyError:
            continue

        end_idx = min(i + 1, len(rebalance_dates) - 1)
        if end_idx <= i:
            continue
        end_date = rebalance_dates[end_idx]

        date_range = returns.index[(returns.index >= date) & (returns.index < end_date)]
        if len(date_range) == 0:
            continue

        common = factor_cross.index.intersection(returns.columns)
        if len(common) < n_groups * 5:
            continue

        ranks = factor_cross[common].rank(pct=True)
        period_ret = returns.loc[date_range, common]

        group_dates.append(date)

        for g in range(1, n_groups + 1):
            lower = (g - 1) / n_groups
            upper = g / n_groups
            if g == 1:
                members = ranks[ranks <= upper].index
            else:
                members = ranks[(ranks > lower) & (ranks <= upper)].index

            if len(members) > 0:
                avg_ret = period_ret[members].mean(axis=1).sum()
            else:
                avg_ret = 0.0
            group_returns[f"G{g}"].append(avg_ret)

    if not group_dates:
        return {"group_stats": pd.DataFrame(), "monotonicity": 0.0}

    group_df = pd.DataFrame(group_returns, index=group_dates)
    group_cum = (1 + group_df).cumprod()

    ls = group_df[f"G{n_groups}"] - group_df["G1"]
    ls_cum = (1 + ls).cumprod()

    periods_per_year = 252 / rebalance_freq
    stats = {}
    for col in group_df.columns:
        s = group_df[col]
        ann_ret = s.mean() * periods_per_year
        ann_vol = s.std() * np.sqrt(periods_per_year) if s.std() > 0 else 0
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        stats[col] = {"ann_return": round(ann_ret, 4),
                      "ann_vol": round(ann_vol, 4),
                      "sharpe": round(sharpe, 3)}

    ls_ann_ret = ls.mean() * periods_per_year
    ls_ann_vol = ls.std() * np.sqrt(periods_per_year) if ls.std() > 0 else 0
    ls_sharpe = ls_ann_ret / ls_ann_vol if ls_ann_vol > 0 else 0
    stats["L-S"] = {"ann_return": round(ls_ann_ret, 4),
                    "ann_vol": round(ls_ann_vol, 4),
                    "sharpe": round(ls_sharpe, 3)}

    group_means = [group_df[f"G{i}"].mean() for i in range(1, n_groups + 1)]
    mono, _ = sp_stats.spearmanr(range(1, n_groups + 1), group_means)

    return {
        "group_stats": pd.DataFrame(stats).T,
        "group_cum_returns": group_cum,
        "long_short_cum": ls_cum,
        "monotonicity": round(mono, 3) if not np.isnan(mono) else 0.0,
    }


def model_backtest(factor_dfs: dict[str, pd.DataFrame],
                   prices: pd.DataFrame,
                   model_name: str,
                   top_n: int = 50,
                   rebalance_freq: int = 21) -> dict:
    """Backtest a multi-factor model (e.g. value, momentum, credit_spread).

    Uses the screener's MODELS weights to score stocks historically,
    then tracks top-N portfolio performance.
    """
    from factor_library.screener import MODELS, score_stocks
    from factor_library.search import build_factor_matrix

    if model_name not in MODELS:
        return {"error": f"Unknown model: {model_name}"}

    if not isinstance(list(factor_dfs.values())[0].index, pd.MultiIndex):
        return {"error": "Factors must have MultiIndex (date, symbol)"}

    first_df = list(factor_dfs.values())[0]
    all_dates = first_df.index.get_level_values(0).unique().sort_values()
    rebalance_dates = all_dates[::rebalance_freq]

    close_pivot = prices.pivot_table(index="date", columns="symbol", values="close")
    returns = close_pivot.pct_change()

    portfolio_returns = []
    portfolio_dates = []

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        date_str = str(date)
        matrix = build_factor_matrix(factor_dfs, date=date_str)
        if matrix.empty or len(matrix) < top_n:
            continue

        scored = score_stocks(matrix, model=model_name, top_n=top_n)
        if scored.empty:
            continue

        selected = scored["symbol"].tolist()

        date_range = returns.index[(returns.index >= date) & (returns.index < next_date)]
        if len(date_range) == 0:
            continue

        available = [s for s in selected if s in returns.columns]
        if not available:
            continue

        period_ret = returns.loc[date_range, available].mean(axis=1).sum()
        portfolio_returns.append(period_ret)
        portfolio_dates.append(date)

    if not portfolio_dates:
        return {"error": "No valid periods"}

    ret_series = pd.Series(portfolio_returns, index=portfolio_dates)
    cum = (1 + ret_series).cumprod()

    periods_per_year = 252 / rebalance_freq
    ann_ret = ret_series.mean() * periods_per_year
    ann_vol = ret_series.std() * np.sqrt(periods_per_year) if ret_series.std() > 0 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    peak = np.maximum.accumulate(cum.values)
    dd = (cum.values - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    return {
        "model": model_name,
        "description": MODELS[model_name]["description"],
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "total_return": round(float(cum.iloc[-1]) - 1, 4) if len(cum) > 0 else 0,
        "n_periods": len(portfolio_returns),
        "cum_returns": cum,
    }


def generate_backtest_report(factor_dfs: dict[str, pd.DataFrame],
                             prices: pd.DataFrame,
                             holding_days: int = 5) -> str:
    """Generate comprehensive factor backtest report."""
    lines = []
    lines.append("=" * 65)
    lines.append("  因子库回测报告")
    lines.append("=" * 65)

    n_stocks = prices["symbol"].nunique()
    date_range = f"{prices['date'].min().strftime('%Y-%m-%d')} ~ " \
                 f"{prices['date'].max().strftime('%Y-%m-%d')}"
    lines.append(f"\n  数据: {n_stocks} 只股票, {date_range}")
    lines.append(f"  前瞻收益: {holding_days} 天")

    # 1. IC Analysis
    lines.append(f"\n{'─' * 65}")
    lines.append("  1. 单因子 IC 分析 (Top 20 by IC_IR)")
    lines.append(f"{'─' * 65}")

    ic_table = run_ic_analysis(factor_dfs, prices, holding_days=holding_days, top_n=20)
    if not ic_table.empty:
        for factor_name, row in ic_table.iterrows():
            ic_ir = row.get("IC_IR", 0)
            quality = "★★★" if ic_ir > 0.5 else ("★★" if ic_ir > 0.3 else "★")
            lines.append(
                f"  {factor_name:15s} | IC={row['IC_mean']:+.4f} | "
                f"IR={ic_ir:.4f} | t={row.get('t_stat', 0):5.1f} | "
                f"正比={row.get('IC_positive_pct', 0):.0%} | "
                f"n={row.get('n_periods', 0):3.0f} | {quality}")

    # 2. Model Backtests
    lines.append(f"\n{'─' * 65}")
    lines.append("  2. 多因子模型回测")
    lines.append(f"{'─' * 65}")

    from factor_library.screener import MODELS
    for model_name in MODELS:
        result = model_backtest(factor_dfs, prices, model_name,
                                top_n=50, rebalance_freq=21)
        if "error" in result:
            lines.append(f"  {model_name:20s} | 错误: {result['error']}")
            continue
        lines.append(
            f"  {model_name:20s} | "
            f"年化={result['ann_return']:+.1%} | "
            f"Sharpe={result['sharpe']:.2f} | "
            f"MaxDD={result['max_drawdown']:.1%} | "
            f"总收益={result['total_return']:+.1%}")

    # 3. Top factor quintile backtests
    lines.append(f"\n{'─' * 65}")
    lines.append("  3. Top 因子五分位回测")
    lines.append(f"{'─' * 65}")

    if not ic_table.empty:
        top_factors = ic_table.head(5)
        for factor_name, row in top_factors.iterrows():
            cat = row.get("category", "")
            if cat in factor_dfs:
                result = quintile_backtest(factor_dfs[cat], prices, factor_name)
                stats = result.get("group_stats")
                mono = result.get("monotonicity", 0)
                if stats is not None and not stats.empty and "L-S" in stats.index:
                    ls = stats.loc["L-S"]
                    lines.append(
                        f"  {factor_name:15s} | "
                        f"L-S年化={ls['ann_return']:+.1%} | "
                        f"Sharpe={ls['sharpe']:.2f} | "
                        f"单调性={mono:+.3f}")

    lines.append(f"\n{'=' * 65}")
    lines.append("  IC_IR > 0.5 = 优秀因子 | Sharpe > 1.0 = 优秀策略")
    lines.append(f"{'=' * 65}")

    return "\n".join(lines)
