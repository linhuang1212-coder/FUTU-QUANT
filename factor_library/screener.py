"""
多因子选股引擎 — 将因子库应用于实战

核心功能:
  1. 多因子打分排序 (自定义权重)
  2. 风控过滤器 (排除高风险标的)
  3. 择时信号 (市场状态判断)
  4. 与现有策略打通 (Credit Spread / 动量轮动)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("factor_library.screener")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 预设因子模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODELS = {
    "value": {
        "description": "价值投资: 低估值 + 高质量",
        "weights": {
            "EP": 0.20, "BP": 0.10, "ROE": 0.20,
            "GROSS_MARGIN": 0.10, "DEBT_EQUITY": -0.10,
            "PIOTROSKI_F": 0.15, "DIV_YIELD": 0.15,
        },
    },
    "momentum": {
        "description": "动量策略: 趋势追踪 + 波动过滤",
        "weights": {
            "MOM_12M_1M": 0.30, "MOM_6M": 0.20,
            "PRICE_SMA200": 0.15, "RSI_14": 0.10,
            "VOL_20D": -0.10, "VOLUME_SURGE": 0.05,
            "TURNOVER": 0.10,
        },
    },
    "quality": {
        "description": "质量策略: 高盈利 + 低杠杆 + 稳定增长",
        "weights": {
            "ROE": 0.25, "GROSS_MARGIN": 0.20,
            "REV_GROWTH": 0.15, "DEBT_EQUITY": -0.15,
            "PIOTROSKI_F": 0.15, "VOL_20D": -0.10,
        },
    },
    "low_risk": {
        "description": "低风险策略: 低波动 + 高Sortino + 低Beta",
        "weights": {
            "VOL_20D": -0.25, "BETA": -0.20,
            "SORTINO": 0.20, "MAX_DD_60D": -0.15,
            "DOWNVOL": -0.10, "CALMAR": 0.10,
        },
    },
    "credit_spread": {
        "description": "Credit Spread 标的筛选: 高IVR + 低风险 + 足够流动性",
        "weights": {
            "IVR": 0.30, "HV_RATIO": 0.15,
            "AMIHUD": -0.15, "SPREAD_PROXY": -0.10,
            "BETA": -0.10, "VOL_20D": 0.10,
            "SORTINO": 0.10,
        },
    },
    "momentum_rotation": {
        "description": "动量轮动 ETF 选择: 强趋势 + 低回撤",
        "weights": {
            "MOM_12M_1M": 0.30, "MOM_6M": 0.20,
            "PRICE_SMA200": 0.15, "MAX_DD_60D": -0.15,
            "VOL_20D": -0.10, "TURNOVER": 0.10,
        },
    },
}


def score_stocks(factor_matrix: pd.DataFrame,
                 model: str = "value",
                 custom_weights: Optional[dict] = None,
                 top_n: int = 20) -> pd.DataFrame:
    """多因子打分排序.

    Args:
        factor_matrix: 截面因子矩阵 (symbols x factors)
        model: 预设模型名 (value/momentum/quality/low_risk/credit_spread)
        custom_weights: 自定义权重 {factor_name: weight}
        top_n: 返回前 N 只

    Returns:
        DataFrame: symbol, score, rank, factor values
    """
    weights = custom_weights or MODELS.get(model, {}).get("weights", {})
    if not weights:
        raise ValueError(f"未知模型: {model}. 可选: {list(MODELS.keys())}")

    # 只用有数据的因子
    available = [f for f in weights if f in factor_matrix.columns]
    if not available:
        logger.warning(f"[Screener] 没有可用因子. 需要: {list(weights.keys())}")
        return pd.DataFrame()

    missing = [f for f in weights if f not in factor_matrix.columns]
    if missing:
        logger.info(f"[Screener] 缺少因子 (跳过): {missing}")

    # 截面排名标准化 [0, 1]
    ranked = factor_matrix[available].rank(pct=True)

    # 加权打分 (NaN 因子贡献 0 分)
    total_weight = sum(abs(weights[f]) for f in available)
    scores = pd.Series(0.0, index=factor_matrix.index)
    for f in available:
        w = weights[f] / total_weight
        col_ranked = ranked[f].fillna(0.5)  # NaN 用中位数
        if w < 0:
            scores += w * (1 - col_ranked)
        else:
            scores += w * col_ranked

    result = pd.DataFrame({
        "symbol": factor_matrix.index,
        "score": scores.values,
    })
    result["rank"] = result["score"].rank(ascending=False, na_option="bottom").fillna(9999).astype(int)
    result = result.sort_values("score", ascending=False)

    # 附加因子值
    for f in available[:5]:
        result[f] = factor_matrix[f].values

    model_desc = MODELS.get(model, {}).get("description", model)
    if not result.empty:
        logger.info(f"[Screener] {model_desc}: {len(available)} 因子, "
                    f"top1={result.iloc[0]['symbol']} (score={result.iloc[0]['score']:.4f})")
    else:
        logger.info(f"[Screener] {model_desc}: 无结果")

    return result.head(top_n).reset_index(drop=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 风控过滤器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def risk_filter(factor_matrix: pd.DataFrame,
                top_pct: float = 0.7,
                ) -> pd.DataFrame:
    """风控过滤: 排除因子最极端的标的 (基于百分位).

    过滤掉波动率/Beta最高、回撤最大的 bottom 30%.
    NaN 值不参与过滤 (保留).
    """
    mask = pd.Series(True, index=factor_matrix.index)

    if "VOL_20D" in factor_matrix.columns:
        ranks = factor_matrix["VOL_20D"].rank(pct=True)
        mask &= (ranks <= top_pct) | ranks.isna()

    if "BETA" in factor_matrix.columns:
        ranks = factor_matrix["BETA"].rank(pct=True)
        mask &= (ranks <= top_pct) | ranks.isna()

    if "MAX_DD_60D" in factor_matrix.columns:
        ranks = factor_matrix["MAX_DD_60D"].rank(pct=True)
        mask &= (ranks >= (1 - top_pct)) | ranks.isna()

    filtered = factor_matrix[mask]
    removed = len(factor_matrix) - len(filtered)
    logger.info(f"[RiskFilter] {len(factor_matrix)} → {len(filtered)} "
                f"(过滤 {removed} 只高风险)")
    return filtered


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 择时信号
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def market_timing_signal(factor_matrix: pd.DataFrame) -> dict:
    """基于全市场因子分布 + HMM regime 判断市场状态.

    HMM regime 权重占 40%，因子信号占 60%。
    HMM 失败时自动 fallback 到纯因子信号。

    Returns:
        dict with market_state, confidence, details
    """
    signals = {}

    # 1. 动量健康度: 多少比例的股票在 SMA200 以上
    if "PRICE_SMA200" in factor_matrix.columns:
        above_sma = (factor_matrix["PRICE_SMA200"] > 0).mean()
        signals["breadth"] = above_sma

    # 2. 波动率环境: 全市场平均 IVR
    if "IVR" in factor_matrix.columns:
        avg_ivr = factor_matrix["IVR"].median()
        signals["ivr_median"] = avg_ivr

    # 3. 动量分散度: 动量因子的标准差 (高 = 分化, 低 = 一致)
    if "MOM_1M" in factor_matrix.columns:
        mom_dispersion = factor_matrix["MOM_1M"].std()
        signals["momentum_dispersion"] = mom_dispersion

    # 4. 下行风险: 平均最大回撤
    if "MAX_DD_60D" in factor_matrix.columns:
        avg_dd = factor_matrix["MAX_DD_60D"].median()
        signals["avg_drawdown"] = avg_dd

    # Factor-based score (60% weight)
    factor_score = 0
    if signals.get("breadth", 0.5) > 0.6:
        factor_score += 1
    elif signals.get("breadth", 0.5) < 0.4:
        factor_score -= 1

    if signals.get("ivr_median", 0.5) < 0.4:
        factor_score += 1
    elif signals.get("ivr_median", 0.5) > 0.7:
        factor_score -= 1

    if signals.get("avg_drawdown", -0.1) > -0.10:
        factor_score += 1
    elif signals.get("avg_drawdown", -0.1) < -0.20:
        factor_score -= 1

    # HMM regime detection (40% weight)
    hmm_regime = None
    hmm_state = None
    hmm_confidence = 0.0
    try:
        from factor_library.regime import get_current_regime
        regime_result = get_current_regime()
        hmm_regime = regime_result.get("regime", "NEUTRAL")
        hmm_state = regime_result.get("market_state", "NEUTRAL")
        hmm_confidence = regime_result.get("confidence", 0)
        signals["hmm_regime"] = hmm_regime
        signals["hmm_confidence"] = hmm_confidence
    except Exception as e:
        logger.debug(f"[HMM] Regime detection failed ({e}), using factors only")

    # Combine: HMM adjusts factor score
    score = factor_score
    if hmm_state == "BULLISH" and hmm_confidence > 0.5:
        score += 1
    elif hmm_state == "BEARISH" and hmm_confidence > 0.5:
        score -= 1

    if score >= 2:
        state = "BULLISH"
    elif score <= -2:
        state = "BEARISH"
    else:
        state = "NEUTRAL"

    recommendations = {
        "BULLISH": "积极做多, 增加动量/成长仓位",
        "NEUTRAL": "均衡配置, 关注质量/价值",
        "BEARISH": "防御为主, 降低仓位, 增加对冲",
    }

    return {
        "market_state": state,
        "score": score,
        "signals": signals,
        "hmm_regime": hmm_regime,
        "recommendation": recommendations.get(state, ""),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略整合接口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_credit_spread_candidates(factor_matrix: pd.DataFrame,
                                 top_n: int = 10) -> pd.DataFrame:
    """为 Credit Spread 策略筛选最佳标的.

    优先: 高 IVR + 低 Beta + 足够流动性
    """
    filtered = risk_filter(factor_matrix, top_pct=0.8)
    if filtered.empty:
        filtered = factor_matrix
    return score_stocks(filtered, model="credit_spread", top_n=top_n)


def get_momentum_candidates(factor_matrix: pd.DataFrame,
                            top_n: int = 5) -> pd.DataFrame:
    """为动量轮动策略筛选标的.

    优先: 强 12M-1M 动量 + 低回撤 + SMA200 以上
    """
    if "PRICE_SMA200" in factor_matrix.columns:
        above_sma = factor_matrix[factor_matrix["PRICE_SMA200"] > 0]
    else:
        above_sma = factor_matrix

    return score_stocks(above_sma, model="momentum_rotation", top_n=top_n)


def generate_daily_report(factor_matrix: pd.DataFrame,
                          sector_map: Optional[dict] = None) -> str:
    """生成每日因子分析报告."""
    lines = []
    lines.append("=" * 50)
    lines.append("  因子库每日报告")
    lines.append("=" * 50)

    # 市场状态
    timing = market_timing_signal(factor_matrix)
    lines.append(f"\n📊 市场状态: {timing['market_state']} (score={timing['score']})")
    for k, v in timing["signals"].items():
        lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    lines.append(f"  建议: {timing['recommendation']}")

    # Top 10 动量股
    lines.append(f"\n🚀 动量 Top 10:")
    mom = score_stocks(factor_matrix, model="momentum", top_n=10)
    if not mom.empty:
        for _, row in mom.iterrows():
            lines.append(f"  {row['rank']:2d}. {row['symbol']:6s} score={row['score']:.4f}")

    # Top 10 价值股
    lines.append(f"\n💎 价值 Top 10:")
    val = score_stocks(factor_matrix, model="value", top_n=10)
    if not val.empty:
        for _, row in val.iterrows():
            lines.append(f"  {row['rank']:2d}. {row['symbol']:6s} score={row['score']:.4f}")

    # Credit Spread 候选
    lines.append(f"\n📈 Credit Spread 候选:")
    cs = get_credit_spread_candidates(factor_matrix, top_n=5)
    if not cs.empty:
        for _, row in cs.iterrows():
            lines.append(f"  {row['symbol']:6s} score={row['score']:.4f}")

    # 异常股票
    from factor_library.search import find_anomalies
    lines.append(f"\n⚠️ 因子异常 Top 5:")
    anomalies = find_anomalies(factor_matrix, top_n=5)
    if not anomalies.empty:
        for _, row in anomalies.iterrows():
            lines.append(f"  {row['symbol']:6s} "
                         f"score={row['anomaly_score']:.2f} "
                         f"({row['extreme_factors']})")

    lines.append(f"\n{'=' * 50}")
    return "\n".join(lines)
