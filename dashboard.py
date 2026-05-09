"""
FUTU-QUANT 可视化仪表盘

启动: streamlit run dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="FUTU-QUANT Dashboard", layout="wide")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=300)
def load_factor_matrix():
    from factor_library.storage import load_factors
    from factor_library.search import build_factor_matrix

    categories = ["technical", "risk", "volatility", "liquidity", "fundamental"]
    factor_dfs = {}
    for cat in categories:
        df = load_factors(cat)
        if not df.empty:
            factor_dfs[cat] = df
    if not factor_dfs:
        return None
    return build_factor_matrix(factor_dfs)


@st.cache_data(ttl=300)
def load_trade_history():
    try:
        from data.trade_store import TradeStore
        store = TradeStore()
        trades = store.query_option_trades(days=30)
        open_trades = store.query_option_trades(status="open")
        store.close()
        return trades, open_trades
    except Exception:
        return [], []


@st.cache_data(ttl=300)
def load_storage_stats():
    try:
        from factor_library.storage import get_storage_stats
        from factor_library.universe import get_universe_stats
        return get_storage_stats(), get_universe_stats()
    except Exception:
        return {}, {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.title("FUTU-QUANT")
page = st.sidebar.radio("", [
    "概览",
    "因子选股",
    "因子热力图",
    "相似股票",
    "市场择时",
    "系统状态",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page: Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if page == "概览":
    st.title("系统概览")

    # Market timing
    matrix = load_factor_matrix()
    if matrix is not None:
        from factor_library.screener import market_timing_signal
        timing = market_timing_signal(matrix)

        state = timing["market_state"]
        color_map = {"BULLISH": "green", "NEUTRAL": "orange", "BEARISH": "red"}
        color = color_map.get(state, "gray")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("市场状态", state)
        col2.metric("择时评分", timing["score"])
        col3.metric("因子覆盖", f"{len(matrix)} 只")
        col4.metric("因子数量", f"{len(matrix.columns)} 个")

        st.info(timing["recommendation"])

    # Open positions
    all_trades, open_trades = load_trade_history()
    st.subheader("当前持仓")
    if open_trades:
        pos_data = []
        for t in open_trades:
            pos_data.append({
                "策略": t.get("strategy", ""),
                "标的": t.get("underlying", ""),
                "最大亏损": f"${t.get('max_loss', 0):.0f}",
                "目标盈利": f"${t.get('target_pnl', 0):.0f}",
                "开仓时间": t.get("timestamp", "")[:16],
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.info("无持仓")

    # Recent P&L
    st.subheader("近 30 天交易")
    closed = [t for t in all_trades if t.get("status") == "closed"]
    if closed:
        pnl_data = []
        for t in closed:
            pnl_data.append({
                "日期": t.get("timestamp", "")[:10],
                "策略": t.get("strategy", ""),
                "标的": t.get("underlying", ""),
                "盈亏": t.get("realized_pnl", 0),
            })
        df = pd.DataFrame(pnl_data)

        total_pnl = df["盈亏"].sum()
        wins = (df["盈亏"] > 0).sum()
        total = len(df)
        win_rate = wins / total * 100 if total > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("总盈亏", f"${total_pnl:+,.2f}")
        c2.metric("胜率", f"{win_rate:.0f}%")
        c3.metric("交易笔数", total)

        # Cumulative P&L chart
        df["累计盈亏"] = df["盈亏"].cumsum()
        st.line_chart(df.set_index("日期")["累计盈亏"])
    else:
        st.info("无近期已平仓交易")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page: Factor Screening
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "因子选股":
    st.title("多因子选股")

    matrix = load_factor_matrix()
    if matrix is None:
        st.error("因子数据不可用, 请先运行 --compute-factors")
    else:
        from factor_library.screener import MODELS, score_stocks, risk_filter

        col1, col2, col3 = st.columns(3)
        model = col1.selectbox("选股模型", list(MODELS.keys()),
                               format_func=lambda m: f"{m} — {MODELS[m]['description']}")
        top_n = col2.number_input("返回数量", 5, 100, 20)
        use_risk_filter = col3.checkbox("启用风控过滤", value=False)

        if use_risk_filter:
            filtered = risk_filter(matrix)
        else:
            filtered = matrix

        results = score_stocks(filtered, model=model, top_n=top_n)
        if not results.empty:
            st.subheader(f"模型: {MODELS[model]['description']}")
            st.dataframe(results, use_container_width=True, hide_index=True)

            # Weight visualization
            weights = MODELS[model]["weights"]
            w_df = pd.DataFrame({
                "因子": list(weights.keys()),
                "权重": list(weights.values()),
            }).sort_values("权重", ascending=True)
            st.bar_chart(w_df.set_index("因子"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page: Factor Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "因子热力图":
    st.title("因子相关性热力图")

    matrix = load_factor_matrix()
    if matrix is None:
        st.error("因子数据不可用")
    else:
        numeric_cols = matrix.select_dtypes(include=[np.number]).columns
        corr = matrix[numeric_cols].corr()

        st.subheader(f"截面相关矩阵 ({len(numeric_cols)} 因子)")

        # Use streamlit's built-in data display
        styled = corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
        st.dataframe(styled, use_container_width=True)

        # High correlation pairs
        st.subheader("高相关因子对 (|r| > 0.7)")
        cols = list(corr.columns)
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.7:
                    pairs.append({"因子 A": cols[i], "因子 B": cols[j],
                                  "相关系数": round(r, 3)})
        if pairs:
            st.dataframe(pd.DataFrame(pairs).sort_values("相关系数",
                         ascending=False, key=abs), hide_index=True)
        else:
            st.info("无高相关因子对")

        # Factor distribution
        st.subheader("因子分布统计")
        stats = matrix[numeric_cols].describe().T
        stats["非空比例"] = (1 - matrix[numeric_cols].isna().mean()).values
        st.dataframe(stats[["count", "mean", "std", "min", "max", "非空比例"]],
                     use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page: Similar Stocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "相似股票":
    st.title("相似股票搜索")

    matrix = load_factor_matrix()
    if matrix is None:
        st.error("因子数据不可用")
    else:
        col1, col2 = st.columns([2, 1])
        symbol = col1.text_input("股票代码", "AAPL").upper()
        top_n = col2.number_input("返回数量", 5, 50, 10)

        if symbol:
            from factor_library.search import find_similar
            results = find_similar(matrix, symbol, top_n=top_n)
            if results.empty:
                st.warning(f"{symbol} 不在因子库中")
            else:
                st.subheader(f"与 {symbol} 最相似的 {top_n} 只股票")
                st.dataframe(results, use_container_width=True, hide_index=True)

                # Show target stock's factor profile
                if symbol in matrix.index:
                    st.subheader(f"{symbol} 因子画像")
                    profile = matrix.loc[symbol].dropna().sort_values()
                    st.bar_chart(profile)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page: Market Timing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "市场择时":
    st.title("市场择时信号")

    matrix = load_factor_matrix()
    if matrix is None:
        st.error("因子数据不可用")
    else:
        from factor_library.screener import market_timing_signal
        timing = market_timing_signal(matrix)

        state = timing["market_state"]
        emoji_map = {"BULLISH": ":green_circle:", "NEUTRAL": ":orange_circle:",
                     "BEARISH": ":red_circle:"}
        emoji = emoji_map.get(state, "")

        st.header(f"{emoji} {state}")
        st.metric("综合评分", timing["score"])

        st.subheader("信号明细")
        for k, v in timing["signals"].items():
            if isinstance(v, float):
                st.metric(k, f"{v:.4f}")
            else:
                st.metric(k, str(v))

        st.info(timing["recommendation"])

        # Market breadth
        st.subheader("市场宽度分析")
        if "PRICE_SMA200" in matrix.columns:
            above = (matrix["PRICE_SMA200"] > 0).mean()
            st.progress(float(above), text=f"{above:.0%} 股票在 SMA200 以上")

        if "MOM_1M" in matrix.columns:
            st.subheader("1个月动量分布")
            mom_data = matrix["MOM_1M"].dropna()
            st.bar_chart(pd.cut(mom_data, bins=20).value_counts().sort_index())

        # Anomaly detection
        st.subheader("因子异常 Top 10")
        from factor_library.search import find_anomalies
        anomalies = find_anomalies(matrix, top_n=10)
        if not anomalies.empty:
            st.dataframe(anomalies, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page: System Status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "系统状态":
    st.title("系统状态")

    storage, universe = load_storage_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("存储")
        if storage:
            st.metric("总大小", f"{storage.get('total_mb', 0):.0f} MB")

            st.caption("行情文件")
            for f in storage.get("parquet_files", []):
                st.text(f"  {f['name']}: {f['size_mb']} MB")

            st.caption("因子文件")
            for f in storage.get("factor_files", []):
                st.text(f"  {f['name']}: {f['size_mb']} MB")
        else:
            st.warning("存储信息不可用")

    with col2:
        st.subheader("Universe")
        if universe:
            st.metric("活跃股票", universe.get("active", 0))
            st.metric("总标的", universe.get("total", 0))

            by_ex = universe.get("by_exchange", {})
            if by_ex:
                st.bar_chart(pd.Series(by_ex))

            meta = universe.get("meta", {})
            if meta.get("last_universe_update"):
                st.text(f"最后更新: {meta['last_universe_update'][:16]}")
        else:
            st.warning("Universe 信息不可用")
