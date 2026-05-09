"""
因子处理器 — 标准化, 去极值, 冗余过滤, PCA

处理流程:
  1. Winsorize: 截断 1%/99% 分位数外的极值
  2. Z-score: 截面标准化 (每天独立)
  3. 冗余过滤: 去除相关系数 > 0.8 的冗余因子
  4. PCA: 降维到解释 90% 方差的主成分
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils.logger import setup_logger

logger = setup_logger("factor_library.processor")


def winsorize(df: pd.DataFrame, limits: tuple = (0.01, 0.01)) -> pd.DataFrame:
    """截面 Winsorize: 每天每个因子独立截断极值."""
    result = df.copy()
    for col in result.columns:
        def _clip_day(x):
            if len(x.dropna()) < 10:
                return x
            lo = x.quantile(limits[0])
            hi = x.quantile(1 - limits[1])
            return x.clip(lo, hi)

        if isinstance(result.index, pd.MultiIndex):
            result[col] = result[col].groupby(level=0).transform(_clip_day)
        else:
            result[col] = _clip_day(result[col])

    return result


def zscore_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """截面 Z-score 标准化: 每天独立计算 (mean=0, std=1)."""
    result = df.copy()
    for col in result.columns:
        def _zscore_day(x):
            m = x.mean()
            s = x.std()
            if s == 0 or pd.isna(s):
                return x * 0
            return (x - m) / s

        if isinstance(result.index, pd.MultiIndex):
            result[col] = result[col].groupby(level=0).transform(_zscore_day)
        else:
            m, s = result[col].mean(), result[col].std()
            if s > 0:
                result[col] = (result[col] - m) / s

    return result


def rank_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """截面百分位排序: 每天独立, 值域 [0, 1]."""
    result = df.copy()
    for col in result.columns:
        if isinstance(result.index, pd.MultiIndex):
            result[col] = result[col].groupby(level=0).transform(
                lambda x: x.rank(pct=True))
        else:
            result[col] = result[col].rank(pct=True)
    return result


def filter_redundant(df: pd.DataFrame, threshold: float = 0.8,
                     ic_values: dict | None = None) -> pd.DataFrame:
    """去除高相关冗余因子.

    对于 |corr| > threshold 的因子对, 保留 IC 更高的; 无 IC 则保留前者.
    """
    corr_matrix = df.corr().abs()
    cols = list(df.columns)
    to_drop = set()

    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr_matrix.loc[cols[i], cols[j]] > threshold:
                if ic_values:
                    ic_i = abs(ic_values.get(cols[i], 0))
                    ic_j = abs(ic_values.get(cols[j], 0))
                    drop = cols[j] if ic_i >= ic_j else cols[i]
                else:
                    drop = cols[j]
                to_drop.add(drop)
                logger.info(f"  冗余: {cols[i]} ↔ {cols[j]} "
                            f"(r={corr_matrix.loc[cols[i], cols[j]]:.3f}) → 移除 {drop}")

    kept = [c for c in cols if c not in to_drop]
    logger.info(f"[Processor] 冗余过滤: {len(cols)} → {len(kept)} 因子 "
                f"(移除 {len(to_drop)})")
    return df[kept]


def pca_reduce(df: pd.DataFrame, variance_threshold: float = 0.9) -> tuple[pd.DataFrame, dict]:
    """PCA 降维到解释指定方差比例的主成分.

    Returns:
        (reduced_df, info_dict with explained_variance_ratio etc.)
    """
    from sklearn.decomposition import PCA

    clean = df.dropna()
    if len(clean) < 10:
        return df, {"error": "too few rows"}

    n_components = min(len(clean.columns), len(clean))
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(clean.values)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_keep = max(1, min(n_keep, n_components))

    cols = [f"PC{i+1}" for i in range(n_keep)]
    result = pd.DataFrame(
        transformed[:, :n_keep],
        index=clean.index,
        columns=cols,
    )

    info = {
        "n_original": len(df.columns),
        "n_components": n_keep,
        "variance_explained": float(cumvar[n_keep - 1]),
        "top_loadings": {},
    }

    for i in range(min(3, n_keep)):
        loadings = pd.Series(pca.components_[i], index=df.columns)
        top = loadings.abs().nlargest(3)
        info["top_loadings"][f"PC{i+1}"] = {
            k: round(float(loadings[k]), 3) for k in top.index
        }

    logger.info(f"[Processor] PCA: {len(df.columns)} 因子 → {n_keep} 主成分 "
                f"(解释方差: {info['variance_explained']:.1%})")
    return result, info


def full_pipeline(df: pd.DataFrame,
                  winsorize_limits: tuple = (0.01, 0.01),
                  zscore: bool = True,
                  remove_redundant: bool = True,
                  redundancy_threshold: float = 0.8,
                  ic_values: dict | None = None,
                  ) -> pd.DataFrame:
    """完整因子处理管道: Winsorize → Z-score → 冗余过滤."""
    logger.info(f"[Processor] 处理管道: {len(df.columns)} 因子, {len(df):,} 行")

    result = winsorize(df, limits=winsorize_limits)

    if zscore:
        result = zscore_cross_sectional(result)

    if remove_redundant and len(result.columns) > 3:
        latest = result
        if isinstance(result.index, pd.MultiIndex):
            last_date = result.index.get_level_values(0).max()
            latest = result.loc[last_date]
        result = result[filter_redundant(latest, threshold=redundancy_threshold,
                                         ic_values=ic_values).columns]

    return result
