"""
数据质量验证 — 拆股检测、缺失值检查、异常值告警
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("factor_library.validator")


def detect_splits(prices: pd.DataFrame, threshold: float = 0.45) -> pd.DataFrame:
    """检测疑似未复权的拆股痕迹.

    Args:
        prices: DataFrame with columns [date, symbol, close]
        threshold: 单日涨跌幅阈值 (默认 45%, 正常股票极少超过)

    Returns:
        DataFrame of suspicious records (symbol, date, return)
    """
    suspicious = []
    for symbol, group in prices.groupby("symbol"):
        g = group.sort_values("date")
        rets = g["close"].pct_change()
        bad = rets[rets.abs() > threshold]
        for idx in bad.index:
            row = g.loc[idx]
            suspicious.append({
                "symbol": symbol,
                "date": row["date"],
                "close": row["close"],
                "return": round(rets[idx], 4),
            })

    if suspicious:
        result = pd.DataFrame(suspicious)
        logger.warning(f"[Validator] 发现 {len(result)} 条疑似未复权记录:")
        for _, r in result.head(10).iterrows():
            logger.warning(f"  {r['symbol']} {r['date']}: return={r['return']:.2%}")
        return result
    else:
        logger.info("[Validator] 拆股检测通过, 未发现异常")
        return pd.DataFrame()


def check_missing_data(prices: pd.DataFrame,
                       max_gap_days: int = 10) -> dict:
    """检查数据缺失情况.

    Returns:
        dict with symbols having too many gaps or too few data points
    """
    issues = {"too_short": [], "large_gaps": [], "stale": []}

    for symbol, group in prices.groupby("symbol"):
        if len(group) < 50:
            issues["too_short"].append({"symbol": symbol, "rows": len(group)})
            continue

        g = group.sort_values("date")
        date_diffs = g["date"].diff().dt.days
        max_gap = date_diffs.max()
        if max_gap and max_gap > max_gap_days:
            issues["large_gaps"].append({
                "symbol": symbol, "max_gap_days": int(max_gap)
            })

        last_date = g["date"].max()
        from datetime import datetime
        days_stale = (datetime.now() - pd.Timestamp(last_date)).days
        if days_stale > 7:
            issues["stale"].append({
                "symbol": symbol, "last_date": str(last_date)[:10],
                "days_stale": days_stale
            })

    total_issues = sum(len(v) for v in issues.values())
    if total_issues > 0:
        logger.warning(f"[Validator] 数据质量问题: "
                       f"太短={len(issues['too_short'])}, "
                       f"大缺口={len(issues['large_gaps'])}, "
                       f"过期={len(issues['stale'])}")
    else:
        logger.info("[Validator] 数据质量检查通过")
    return issues


def check_outliers(prices: pd.DataFrame,
                   volume_z_threshold: float = 10.0) -> list[dict]:
    """检测价格和成交量异常值."""
    outliers = []
    for symbol, group in prices.groupby("symbol"):
        g = group.sort_values("date")
        if len(g) < 20:
            continue

        if "volume" in g.columns:
            vol = g["volume"].astype(float)
            vol_mean = vol.rolling(20, min_periods=10).mean()
            vol_std = vol.rolling(20, min_periods=10).std()
            vol_z = ((vol - vol_mean) / vol_std.replace(0, np.nan)).abs()
            extreme_vol = vol_z[vol_z > volume_z_threshold]
            for idx in extreme_vol.head(3).index:
                row = g.loc[idx]
                outliers.append({
                    "symbol": symbol,
                    "date": str(row["date"])[:10],
                    "type": "volume_spike",
                    "value": float(row["volume"]),
                    "z_score": round(float(vol_z[idx]), 1),
                })

    if outliers:
        logger.info(f"[Validator] 发现 {len(outliers)} 个异常值")
    return outliers


def generate_quality_report(prices: pd.DataFrame) -> str:
    """生成完整的数据质量报告."""
    n_symbols = prices["symbol"].nunique()
    n_rows = len(prices)
    date_range = f"{prices['date'].min()} to {prices['date'].max()}"

    splits = detect_splits(prices)
    missing = check_missing_data(prices)
    outliers = check_outliers(prices)

    lines = [
        "=" * 60,
        "  数据质量报告",
        "=" * 60,
        f"  标的数: {n_symbols}",
        f"  总行数: {n_rows:,}",
        f"  日期范围: {date_range}",
        "",
        f"  拆股/异常跳变: {len(splits)} 条",
        f"  数据太短 (<50行): {len(missing['too_short'])} 只",
        f"  大缺口 (>10天): {len(missing['large_gaps'])} 只",
        f"  数据过期 (>7天): {len(missing['stale'])} 只",
        f"  异常值: {len(outliers)} 个",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print(report)
    return report
