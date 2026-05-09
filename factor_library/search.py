"""
向量检索引擎 — 股票相似性搜索 + 聚类 + 异常检测

Phase 3a: 纯 numpy/scipy 实现 (0 额外依赖)
- 相似股票搜索 (cosine similarity)
- K-Means 聚类 (风格分类)
- 异常检测 (Z-score 离群)
- 历史截面对比

后续可升级到 ChromaDB (Phase 3b) 如果需要历史快照搜索
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from utils.logger import setup_logger

logger = setup_logger("factor_library.search")


def build_factor_matrix(factor_dfs: dict[str, pd.DataFrame],
                        date: Optional[str] = None) -> pd.DataFrame:
    """合并多类因子为一个截面矩阵 (symbols x factors).

    Args:
        factor_dfs: {"technical": df, "risk": df, ...} — 每个 df 索引为 (date, symbol)
        date: 指定日期 (默认最新)

    Returns:
        DataFrame: index=symbol, columns=factor names, values=z-score
    """
    parts = []

    for category, df in factor_dfs.items():
        if df.empty:
            continue
        if isinstance(df.index, pd.MultiIndex):
            if date is None:
                date = str(df.index.get_level_values(0).max())
            try:
                cross = df.loc[pd.Timestamp(date)]
            except KeyError:
                dates = df.index.get_level_values(0).unique()
                closest = dates[dates <= pd.Timestamp(date)]
                if len(closest) == 0:
                    continue
                cross = df.loc[closest[-1]]
            parts.append(cross)
        else:
            parts.append(df)

    if not parts:
        return pd.DataFrame()

    merged = pd.concat(parts, axis=1)
    merged = merged.loc[merged.index.dropna()]
    return merged


def find_similar(matrix: pd.DataFrame, symbol: str,
                 top_n: int = 10,
                 exclude_same_sector: bool = False,
                 sector_map: Optional[dict] = None) -> pd.DataFrame:
    """找和指定股票因子特征最相似的 N 只股票.

    Args:
        matrix: factor matrix (symbols x factors)
        symbol: target stock ticker
        top_n: number of similar stocks to return
        exclude_same_sector: whether to exclude same sector
        sector_map: {symbol: sector} mapping

    Returns:
        DataFrame with columns: symbol, similarity, distance
    """
    if symbol not in matrix.index:
        logger.warning(f"[Search] {symbol} 不在因子矩阵中")
        return pd.DataFrame()

    filled = matrix.fillna(0)
    target = filled.loc[symbol].values.reshape(1, -1)
    others = filled.drop(symbol, errors="ignore")

    dists = cdist(target, others.values, metric="cosine")[0]
    similarities = 1 - dists

    result = pd.DataFrame({
        "symbol": others.index,
        "similarity": similarities,
        "distance": dists,
    }).sort_values("similarity", ascending=False)

    if exclude_same_sector and sector_map:
        target_sector = sector_map.get(symbol, "")
        if target_sector:
            result = result[result["symbol"].map(
                lambda s: sector_map.get(s, "") != target_sector)]

    return result.head(top_n).reset_index(drop=True)


def find_anomalies(matrix: pd.DataFrame,
                   threshold: float = 3.0,
                   top_n: int = 20) -> pd.DataFrame:
    """检测因子异常股票 (多维 Z-score 离群).

    使用每只股票的因子向量到市场均值的 Mahalanobis-like 距离.
    简化为各因子 Z-score 绝对值之和.
    """
    filled = matrix.fillna(0)

    # 截面 Z-score
    zscored = (filled - filled.mean()) / filled.std().replace(0, 1)
    anomaly_score = zscored.abs().mean(axis=1)

    result = pd.DataFrame({
        "symbol": anomaly_score.index,
        "anomaly_score": anomaly_score.values,
    }).sort_values("anomaly_score", ascending=False)

    # 找出每只异常股票的最极端因子
    top_factors = []
    for sym in result.head(top_n)["symbol"]:
        row = zscored.loc[sym]
        extreme = row.abs().nlargest(3)
        factors_str = ", ".join(
            f"{f}={row[f]:+.2f}" for f in extreme.index)
        top_factors.append(factors_str)

    result_top = result.head(top_n).copy()
    result_top["extreme_factors"] = top_factors
    return result_top.reset_index(drop=True)


def cluster_stocks(matrix: pd.DataFrame,
                   n_clusters: int = 8) -> pd.DataFrame:
    """K-Means 聚类 — 将股票按因子特征分组.

    Returns:
        DataFrame: symbol, cluster, distance_to_center
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    filled = matrix.fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled.values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    distances = np.min(kmeans.transform(scaled), axis=1)

    result = pd.DataFrame({
        "symbol": filled.index,
        "cluster": labels,
        "distance_to_center": distances,
    })

    # 每个聚类的特征描述
    cluster_profiles = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_mean = filled.values[mask].mean(axis=0)
        top_idx = np.argsort(np.abs(cluster_mean))[-3:][::-1]
        profile = ", ".join(
            f"{filled.columns[i]}={cluster_mean[i]:+.2f}" for i in top_idx)
        cluster_profiles.append(profile)
        count = mask.sum()
        logger.info(f"  Cluster {c} ({count} 只): {profile}")

    return result.reset_index(drop=True)


def compare_to_history(current_matrix: pd.DataFrame,
                       historical_matrices: dict[str, pd.DataFrame],
                       metric: str = "cosine") -> pd.DataFrame:
    """比较当前市场截面和历史截面的相似度.

    Args:
        current_matrix: 当前因子矩阵 (symbols x factors)
        historical_matrices: {date_str: matrix} 历史截面
        metric: 距离度量

    Returns:
        DataFrame: date, similarity, market_state
    """
    common_symbols = list(current_matrix.index)
    current_mean = current_matrix.loc[common_symbols].fillna(0).mean().values.reshape(1, -1)

    results = []
    for date_str, hist_matrix in historical_matrices.items():
        common = list(set(common_symbols) & set(hist_matrix.index))
        if len(common) < 100:
            continue
        hist_mean = hist_matrix.loc[common].fillna(0).mean().values.reshape(1, -1)

        if current_mean.shape[1] != hist_mean.shape[1]:
            min_cols = min(current_mean.shape[1], hist_mean.shape[1])
            dist = cdist(current_mean[:, :min_cols],
                         hist_mean[:, :min_cols], metric=metric)[0, 0]
        else:
            dist = cdist(current_mean, hist_mean, metric=metric)[0, 0]

        results.append({
            "date": date_str,
            "similarity": 1 - dist,
            "distance": dist,
        })

    return pd.DataFrame(results).sort_values("similarity", ascending=False)
