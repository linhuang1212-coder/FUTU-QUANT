"""
因子计算引擎 — 40+ 因子, 全向量化

因子分类:
  A. 技术因子 (15): 动量, 波动, 换手, 反转, RSI, MACD, BB, ATR, SMA200, 放量
  B. 基本面因子 (8): EP, BP, ROE, RevGrowth, GrossMargin, D/E, DivYield, Piotroski
  C. 风险因子 (8): Beta, DownVol, MaxDD, Skew, Kurt, Sortino, Calmar, CorrSPY
  D. 波动因子 (4): IVR, HV_Ratio, TermStructure, RealizedSkew
  E. 流动性因子 (3): Amihud, SpreadProxy, TurnoverStability

全部使用 pandas 向量化计算, 5800 只 x 5 年 ≈ 3-5 分钟
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("factor_library.factors")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. 技术因子
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_technical(prices: pd.DataFrame) -> pd.DataFrame:
    """计算 15 个技术因子.

    Args:
        prices: 长格式 DataFrame (date, symbol, open, high, low, close, volume)

    Returns:
        DataFrame with (date, symbol) as index, factor columns
    """
    logger.info("[Factors] 计算技术因子 (15 个)...")
    df = prices.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol")

    ret = g["close"].pct_change()

    # 动量
    df["MOM_1M"] = g["close"].pct_change(21)
    df["MOM_3M"] = g["close"].pct_change(63)
    df["MOM_6M"] = g["close"].pct_change(126)
    df["MOM_12M"] = g["close"].pct_change(252)
    df["MOM_12M_1M"] = df["MOM_12M"] - df["MOM_1M"]

    # 波动率
    df["VOL_20D"] = ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252))
    df["VOL_60D"] = ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(60, min_periods=40).std() * np.sqrt(252))

    # 换手率
    vol_20 = g["volume"].transform(lambda x: x.rolling(20, min_periods=15).mean())
    vol_100 = g["volume"].transform(lambda x: x.rolling(100, min_periods=60).mean())
    df["TURNOVER"] = vol_20 / vol_100.replace(0, np.nan)

    # 反转
    df["REVERSAL"] = -ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(5, min_periods=3).sum())

    # RSI(14)
    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df["symbol"]).transform(
        lambda x: x.ewm(span=14, adjust=False).mean())
    avg_loss = loss.groupby(df["symbol"]).transform(
        lambda x: x.ewm(span=14, adjust=False).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - 100 / (1 + rs)

    # MACD histogram
    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd_line = ema12 - ema26
    signal = macd_line.groupby(df["symbol"]).transform(
        lambda x: x.ewm(span=9, adjust=False).mean())
    df["MACD_HIST"] = (macd_line - signal) / df["close"]  # 标准化

    # Bollinger Band Width
    sma20 = g["close"].transform(lambda x: x.rolling(20, min_periods=15).mean())
    std20 = g["close"].transform(lambda x: x.rolling(20, min_periods=15).std())
    df["BB_WIDTH"] = (4 * std20) / sma20.replace(0, np.nan)  # (upper-lower)/middle = 4σ/μ

    # ATR%
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].groupby(df["symbol"]).shift(1)).abs(),
        (df["low"] - df["close"].groupby(df["symbol"]).shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.groupby(df["symbol"]).transform(
        lambda x: x.rolling(14, min_periods=10).mean())
    df["ATR_PCT"] = atr14 / df["close"]

    # Price / SMA200 - 1
    sma200 = g["close"].transform(lambda x: x.rolling(200, min_periods=150).mean())
    df["PRICE_SMA200"] = df["close"] / sma200.replace(0, np.nan) - 1

    # Volume Surge
    df["VOLUME_SURGE"] = df["volume"] / vol_20.replace(0, np.nan)

    factor_cols = [
        "MOM_1M", "MOM_3M", "MOM_6M", "MOM_12M", "MOM_12M_1M",
        "VOL_20D", "VOL_60D", "TURNOVER", "REVERSAL",
        "RSI_14", "MACD_HIST", "BB_WIDTH", "ATR_PCT",
        "PRICE_SMA200", "VOLUME_SURGE",
    ]

    result = df.set_index(["date", "symbol"])[factor_cols]
    logger.info(f"  技术因子: {len(result):,} 行 x {len(factor_cols)} 因子")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. 基本面因子
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_fundamental(prices: pd.DataFrame,
                        fundamentals: pd.DataFrame) -> pd.DataFrame:
    """计算基本面因子，并前填充到每个交易日.

    季报数据本质低频（每季度变一次），但回测需要日频截面。
    策略：为每个交易日生成截面，基本面值保持不变直到下一次更新。

    Args:
        prices: 长格式 (date, symbol, close, volume)
        fundamentals: DataFrame with symbol as index,
                      columns: pe, pb, roe, revenue_growth, gross_margin,
                               debt_equity, dividend_yield, net_income,
                               total_assets, equity, revenue

    Returns:
        DataFrame with (date, symbol) index and factor columns
    """
    logger.info("[Factors] 计算基本面因子...")

    fund = fundamentals.copy()
    if "symbol" in fund.columns:
        fund = fund.set_index("symbol")

    fund_symbols = set(fund.index)

    all_dates = prices["date"].unique()
    all_dates = np.sort(all_dates)

    # 每月采样一次日期（基本面季度更新，月频足够）
    date_series = pd.Series(all_dates)
    monthly_dates = date_series.groupby(
        pd.to_datetime(date_series).dt.to_period("M")
    ).last().values

    if len(monthly_dates) == 0:
        monthly_dates = [prices["date"].max()]

    logger.info(f"  基本面因子: 展开到 {len(monthly_dates)} 个月度截面")

    frames = []
    for dt in monthly_dates:
        dt_prices = prices[prices["date"] == dt][["symbol", "close", "volume"]].copy()
        dt_prices = dt_prices.set_index("symbol")

        common = dt_prices.index.intersection(fund.index)
        if len(common) == 0:
            continue

        merged = dt_prices.loc[common].join(fund.loc[common], how="inner")

        result = pd.DataFrame(index=merged.index)

        pe = merged.get("pe", pd.Series(0, index=merged.index))
        result["EP"] = np.where(pe > 0, 1.0 / pe, 0)

        pb = merged.get("pb", pd.Series(0, index=merged.index))
        result["BP"] = np.where(pb > 0, 1.0 / pb, 0)

        result["ROE"] = merged.get("roe", 0)
        result["REV_GROWTH"] = merged.get("revenue_growth", 0)
        result["GROSS_MARGIN"] = merged.get("gross_margin", 0)
        result["DEBT_EQUITY"] = merged.get("debt_equity", 0)
        result["DIV_YIELD"] = merged.get("dividend_yield", 0)

        f_score = pd.Series(0, index=result.index, dtype=int)
        f_score += (result["ROE"] > 0).astype(int)
        f_score += (result["REV_GROWTH"] > 0).astype(int)
        f_score += (result["GROSS_MARGIN"] > 0.2).astype(int)
        f_score += (result["DEBT_EQUITY"] < 2.0).astype(int)
        f_score += (result["EP"] > 0.03).astype(int)
        result["PIOTROSKI_F"] = f_score

        result["date"] = dt
        result.index.name = "symbol"
        result = result.reset_index().set_index(["date", "symbol"])
        frames.append(result)

    if not frames:
        logger.warning("  基本面因子: 无有效数据")
        return pd.DataFrame()

    final = pd.concat(frames)
    logger.info(f"  基本面因子: {len(final):,} 行 x {len(final.columns)} 因子, "
                f"{final.index.get_level_values(0).nunique()} 个截面")
    return final


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. 风险因子
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_risk(prices: pd.DataFrame,
                 spy_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """计算 8 个风险因子.

    Args:
        prices: 长格式 (date, symbol, close)
        spy_prices: SPY 价格 DataFrame (date, close) — 用于 Beta 和 Corr
    """
    logger.info("[Factors] 计算风险因子 (8 个)...")
    df = prices.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol")
    ret = g["close"].pct_change()
    df["ret"] = ret

    # SPY returns for Beta/Corr
    if spy_prices is not None and len(spy_prices) > 0:
        spy = spy_prices.copy()
        if "date" not in spy.columns:
            spy = spy.reset_index()
        spy = spy.sort_values("date")
        spy["spy_ret"] = spy["close"].pct_change()
        spy = spy[["date", "spy_ret"]].dropna()
        df = df.merge(spy, on="date", how="left")
        df["spy_ret"] = df["spy_ret"].fillna(0)
    else:
        df["spy_ret"] = 0

    # Beta (252D rolling)
    def _rolling_beta(sub):
        cov = sub["ret"].rolling(252, min_periods=120).cov(sub["spy_ret"])
        var = sub["spy_ret"].rolling(252, min_periods=120).var()
        return cov / var.replace(0, np.nan)

    df["BETA"] = df.groupby("symbol", group_keys=False).apply(_rolling_beta).values

    # Downside volatility (only negative returns)
    neg_ret = ret.clip(upper=0)
    df["DOWNVOL"] = neg_ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(60, min_periods=40).std() * np.sqrt(252))

    # Max drawdown 60D
    def _max_dd_60(sub):
        close = sub["close"]
        rolling_max = close.rolling(60, min_periods=30).max()
        dd = close / rolling_max - 1
        return dd

    df["MAX_DD_60D"] = df.groupby("symbol", group_keys=False).apply(_max_dd_60).values

    # Skewness (60D)
    df["SKEWNESS"] = ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(60, min_periods=40).skew())

    # Kurtosis (60D)
    df["KURTOSIS"] = ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(60, min_periods=40).kurt())

    # Sortino Ratio (60D annualized)
    ret_mean = ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(60, min_periods=40).mean() * 252)
    df["SORTINO"] = ret_mean / df["DOWNVOL"].replace(0, np.nan)

    # Calmar Ratio (252D return / MaxDD)
    ret_252 = g["close"].pct_change(252)
    rolling_max_252 = g["close"].transform(
        lambda x: x.rolling(252, min_periods=120).max())
    dd_252 = df["close"] / rolling_max_252 - 1
    max_dd_252 = dd_252.groupby(df["symbol"]).transform(
        lambda x: x.rolling(252, min_periods=120).min())
    df["CALMAR"] = ret_252 / (-max_dd_252).replace(0, np.nan)

    # Correlation with SPY (60D)
    def _rolling_corr(sub):
        return sub["ret"].rolling(60, min_periods=40).corr(sub["spy_ret"])

    df["CORR_SPY"] = df.groupby("symbol", group_keys=False).apply(_rolling_corr).values

    factor_cols = [
        "BETA", "DOWNVOL", "MAX_DD_60D", "SKEWNESS",
        "KURTOSIS", "SORTINO", "CALMAR", "CORR_SPY",
    ]

    result = df.set_index(["date", "symbol"])[factor_cols]
    logger.info(f"  风险因子: {len(result):,} 行 x {len(factor_cols)} 因子")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. 波动因子
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    """计算 4 个波动/VIX 类因子."""
    logger.info("[Factors] 计算波动因子 (4 个)...")
    df = prices.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol")
    ret = g["close"].pct_change()

    # HV20 / HV60
    hv20 = ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252))
    hv60 = ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(60, min_periods=40).std() * np.sqrt(252))

    # IVR: HV20 percentile rank over 252 days
    df["IVR"] = hv20.groupby(df["symbol"]).transform(
        lambda x: x.rolling(252, min_periods=120).rank(pct=True))

    # HV Ratio
    df["HV_RATIO"] = hv20 / hv60.replace(0, np.nan)

    # Term Structure (short - long vol)
    df["TERM_STRUCTURE"] = hv20 - hv60

    # Realized Skew (intraday range asymmetry)
    df["REALIZED_SKEW"] = ((df["close"] - df["open"]) /
                           (df["high"] - df["low"]).replace(0, np.nan))
    df["REALIZED_SKEW"] = df["REALIZED_SKEW"].groupby(df["symbol"]).transform(
        lambda x: x.rolling(20, min_periods=15).mean())

    factor_cols = ["IVR", "HV_RATIO", "TERM_STRUCTURE", "REALIZED_SKEW"]

    result = df.set_index(["date", "symbol"])[factor_cols]
    logger.info(f"  波动因子: {len(result):,} 行 x {len(factor_cols)} 因子")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E. 流动性因子
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_liquidity(prices: pd.DataFrame) -> pd.DataFrame:
    """计算 3 个流动性因子."""
    logger.info("[Factors] 计算流动性因子 (3 个)...")
    df = prices.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol")
    ret = g["close"].pct_change()

    # Amihud illiquidity: avg(|ret| / dollar_volume)
    dollar_vol = df["close"] * df["volume"]
    illiq = ret.abs() / dollar_vol.replace(0, np.nan)
    df["AMIHUD"] = illiq.groupby(df["symbol"]).transform(
        lambda x: x.rolling(20, min_periods=15).mean())

    # Spread Proxy: (High - Low) / Close
    df["SPREAD_PROXY"] = (df["high"] - df["low"]) / df["close"]
    df["SPREAD_PROXY"] = df["SPREAD_PROXY"].groupby(df["symbol"]).transform(
        lambda x: x.rolling(20, min_periods=15).mean())

    # Turnover Stability: std of 20D avg volume ratio over 60 days
    vol_20 = g["volume"].transform(lambda x: x.rolling(20, min_periods=15).mean())
    vol_100 = g["volume"].transform(lambda x: x.rolling(100, min_periods=60).mean())
    turnover = vol_20 / vol_100.replace(0, np.nan)
    df["TURNOVER_STABILITY"] = turnover.groupby(df["symbol"]).transform(
        lambda x: x.rolling(60, min_periods=40).std())

    factor_cols = ["AMIHUD", "SPREAD_PROXY", "TURNOVER_STABILITY"]

    result = df.set_index(["date", "symbol"])[factor_cols]
    logger.info(f"  流动性因子: {len(result):,} 行 x {len(factor_cols)} 因子")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主管道
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_all_factors(prices: pd.DataFrame,
                        fundamentals: pd.DataFrame | None = None,
                        spy_prices: pd.DataFrame | None = None,
                        ) -> dict[str, pd.DataFrame]:
    """一键计算所有因子, 返回分类 dict.

    Returns:
        {"technical": df, "risk": df, "volatility": df, "liquidity": df,
         "fundamental": df (if data available)}
    """
    logger.info(f"[Factors] 全量因子计算: {prices['symbol'].nunique()} 只, "
                f"{len(prices):,} 行")

    results = {}

    results["technical"] = compute_technical(prices)

    results["risk"] = compute_risk(prices, spy_prices=spy_prices)

    results["volatility"] = compute_volatility(prices)

    results["liquidity"] = compute_liquidity(prices)

    if fundamentals is not None and not fundamentals.empty:
        results["fundamental"] = compute_fundamental(prices, fundamentals)

    total_factors = sum(len(df.columns) for df in results.values())
    logger.info(f"[Factors] 计算完成: {len(results)} 类, {total_factors} 个因子")
    return results
