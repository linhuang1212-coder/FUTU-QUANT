"""
美股因子库 — 一键构建和更新

Usage:
  python run_factor_library.py --init              # 首次: 下载 ticker + 全量行情 + 基本面
  python run_factor_library.py --update            # 每日: 增量更新行情
  python run_factor_library.py --fundamentals      # 季度: 更新基本面数据
  python run_factor_library.py --compute-factors   # 全量计算因子 (38个)
  python run_factor_library.py --stats             # 查看状态
  python run_factor_library.py --validate          # 数据质量检查
"""
from __future__ import annotations

import sys
import io
import argparse
import time
from datetime import datetime

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def cmd_init(args):
    """首次初始化: 下载全部数据."""
    from factor_library.universe import refresh_universe, get_active_symbols
    from factor_library.pipeline import download_prices_batch, download_fundamentals
    from factor_library.storage import save_prices
    from factor_library.validator import detect_splits, generate_quality_report

    print("=" * 60)
    print("  Phase 1: 初始化美股因子库")
    print("=" * 60)

    # Step 1: 获取 ticker 列表
    print("\n[Step 1/5] 获取全美股 ticker 列表...")
    result = refresh_universe()
    print(f"  原始: {result.get('total', 0)} | "
          f"过滤后: {result.get('filtered', 0)} | "
          f"入库: {result.get('saved', 0)}")

    # Step 2: 获取活跃 ticker
    symbols = get_active_symbols()
    if not symbols:
        print("  ERROR: 无法获取 ticker 列表")
        return

    # 限制范围 (可选)
    if args.limit:
        symbols = symbols[:args.limit]
        print(f"  限制为前 {args.limit} 只 (--limit)")

    print(f"\n[Step 2/5] 准备下载 {len(symbols)} 只股票行情...")

    # Step 3: 批量下载行情
    start_time = time.time()
    prices = download_prices_batch(symbols, years=args.years, resume=not args.fresh)
    elapsed = time.time() - start_time

    if prices.empty:
        print("  WARNING: 未下载到新数据 (可能已全部完成)")
        return

    print(f"  下载耗时: {elapsed / 60:.1f} 分钟")
    print(f"  数据量: {prices['symbol'].nunique()} 只, {len(prices):,} 行")

    # Step 4: 保存到 Parquet
    print(f"\n[Step 3/5] 保存到 Parquet (按年分区)...")
    save_prices(prices)

    # Step 4: 数据质量验证
    print(f"\n[Step 4/5] 数据质量检查...")
    generate_quality_report(prices)

    # Step 5: 基本面 (SEC EDGAR, 完全免费)
    if not args.skip_fundamentals:
        print(f"\n[Step 5/5] SEC EDGAR 基本面采集...")
        downloaded_symbols = prices["symbol"].unique().tolist()
        # 临时设置 skip_sectors=True 避免首次初始化太慢
        args_copy = argparse.Namespace(**vars(args))
        args_copy.skip_sectors = True
        cmd_fundamentals(args_copy)
    else:
        print("\n[Step 5/5] 跳过基本面 (--skip-fundamentals)")

    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  初始化完成! 总耗时: {total_elapsed / 60:.1f} 分钟")
    print(f"{'=' * 60}")

    cmd_stats(args)


def cmd_update(args):
    """每日增量更新."""
    from factor_library.universe import get_active_symbols
    from factor_library.pipeline import incremental_update
    from factor_library.storage import save_prices

    symbols = get_active_symbols()
    if not symbols:
        print("Universe 为空, 请先运行 --init")
        return

    print(f"[Update] 增量更新 {len(symbols)} 只...")
    start = time.time()
    prices = incremental_update(symbols)

    if prices.empty:
        print("  无新数据")
        return

    save_prices(prices)
    elapsed = time.time() - start
    print(f"  更新完成: {prices['symbol'].nunique()} 只, "
          f"{len(prices):,} 行, 耗时 {elapsed:.1f} 秒")


def cmd_fundamentals(args):
    """更新基本面数据 (SEC EDGAR frames API)."""
    from factor_library.universe import get_active_symbols, batch_update_info, DB_PATH, _init_db
    from factor_library.pipeline import download_fundamentals_sec, refresh_cik_map, download_sector_info
    import sqlite3

    symbols = get_active_symbols()
    if not symbols:
        print("Universe 为空, 请先运行 --init")
        return

    if args.limit:
        symbols = symbols[:args.limit]

    # Step 1: CIK 映射
    print(f"\n[Step 1/3] 下载 SEC ticker-CIK 映射...")
    cik_map = refresh_cik_map()
    print(f"  CIK 映射: {len(cik_map)} 家公司")

    # Step 2: 基本面数据 (frames API, ~24 次调用)
    print(f"\n[Step 2/3] SEC EDGAR 基本面采集 ({len(symbols)} 只)...")
    fund_start = time.time()
    fundamentals = download_fundamentals_sec(symbols, cik_map=cik_map)
    fund_elapsed = time.time() - fund_start
    print(f"  基本面耗时: {fund_elapsed:.1f} 秒")
    print(f"  获取成功: {len(fundamentals)} 只")

    if fundamentals:
        batch_update_info(fundamentals)

        conn = sqlite3.connect(str(DB_PATH))
        _init_db(conn)
        now = datetime.now().strftime("%Y-%m-%d")
        for f in fundamentals:
            conn.execute("""
                INSERT OR REPLACE INTO fundamentals
                (symbol, report_date, pe, pb, roe, revenue_growth,
                 gross_margin, debt_equity, dividend_yield)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (f["symbol"], now, f.get("pe", 0), f.get("pb", 0),
                  f.get("roe", 0), f.get("revenue_growth", 0),
                  f.get("gross_margin", 0), f.get("debt_equity", 0),
                  f.get("dividend_yield", 0)))
        conn.commit()
        conn.close()
        print(f"  已写入 SQLite: {len(fundamentals)} 条")

    # Step 3: 行业分类 (可选, 较慢)
    if not args.skip_sectors:
        print(f"\n[Step 3/3] SEC 行业分类采集...")
        # 只更新还没有行业信息的
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.execute(
            "SELECT symbol FROM universe WHERE is_active=1 AND (sector='' OR sector IS NULL)")
        no_sector = [row[0] for row in cur.fetchall()]
        conn.close()

        if no_sector:
            print(f"  待更新行业分类: {len(no_sector)} 只")
            sectors = download_sector_info(no_sector[:2000], cik_map=cik_map)
            if sectors:
                conn = sqlite3.connect(str(DB_PATH))
                _init_db(conn)
                for s in sectors:
                    conn.execute("""
                        UPDATE universe SET sector=?, industry=?, last_updated=?
                        WHERE symbol=?
                    """, (s["sector"], s["industry"],
                          datetime.now().isoformat(), s["symbol"]))
                conn.commit()
                conn.close()
                print(f"  行业分类更新: {len(sectors)} 只")
        else:
            print("  行业分类已全部填充")
    else:
        print("\n[Step 3/3] 跳过行业分类 (--skip-sectors)")


def cmd_stats(args):
    """显示因子库状态."""
    from factor_library.universe import get_universe_stats
    from factor_library.storage import get_storage_stats

    print("\n" + "=" * 60)
    print("  美股因子库状态")
    print("=" * 60)

    u = get_universe_stats()
    print(f"\n  Universe:")
    print(f"    总标的: {u.get('total', 0)}")
    print(f"    活跃:   {u.get('active', 0)}")
    for ex, cnt in u.get("by_exchange", {}).items():
        print(f"    {ex}: {cnt}")
    meta = u.get("meta", {})
    if meta.get("last_universe_update"):
        print(f"    最后更新: {meta['last_universe_update'][:16]}")

    s = get_storage_stats()
    print(f"\n  存储:")
    for f in s.get("parquet_files", []):
        print(f"    {f['name']}: {f['size_mb']} MB")
    for f in s.get("factor_files", []):
        print(f"    {f['name']}: {f['size_mb']} MB")
    print(f"    总计: {s.get('total_mb', 0)} MB")

    print("=" * 60)


def cmd_compute_factors(args):
    """全量计算因子 (38个)."""
    import numpy as np
    import pandas as pd
    from factor_library.storage import load_prices, save_factors
    from factor_library.factors import compute_all_factors
    from factor_library.processor import full_pipeline
    import sqlite3

    print("=" * 60)
    print("  Phase 2: 因子计算")
    print("=" * 60)

    # 加载价格数据
    print("\n[Step 1/4] 加载价格数据...")
    load_start = time.time()
    prices = load_prices()
    if prices.empty:
        print("  无数据, 请先运行 --init")
        return
    print(f"  {prices['symbol'].nunique()} 只, {len(prices):,} 行 "
          f"({time.time() - load_start:.1f}s)")

    # 加载 SPY 价格 (用于 Beta/Corr 计算)
    print("\n[Step 2/4] 加载 SPY 基准数据...")
    spy = None
    try:
        spy_in_prices = prices[prices["symbol"] == "SPY"]
        if len(spy_in_prices) > 100:
            spy = spy_in_prices[["date", "close"]].copy()
            print(f"  SPY (从本地数据): {len(spy)} 天")
        else:
            import yfinance as yf
            spy_data = yf.download("SPY", period="6y", progress=False, auto_adjust=True)
            if spy_data is not None and not spy_data.empty:
                spy = spy_data.reset_index()
                spy.columns = [str(c).lower() if isinstance(c, str) else str(c[0]).lower()
                               for c in spy.columns]
                if "date" not in spy.columns and "index" in spy.columns:
                    spy = spy.rename(columns={"index": "date"})
                spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None)
                spy = spy[["date", "close"]].dropna()
                print(f"  SPY (yfinance): {len(spy)} 天")
    except Exception as e:
        print(f"  SPY 加载失败 ({e}), Beta/Corr 将使用默认值")

    # 加载基本面
    fundamentals = None
    try:
        from factor_library.universe import DB_PATH
        conn = sqlite3.connect(str(DB_PATH))
        fundamentals = pd.read_sql(
            "SELECT * FROM fundamentals WHERE report_date = "
            "(SELECT MAX(report_date) FROM fundamentals)", conn)
        conn.close()
        if not fundamentals.empty:
            print(f"  基本面: {len(fundamentals)} 只")
    except Exception:
        pass

    # 计算因子
    print("\n[Step 3/4] 计算全部因子...")
    calc_start = time.time()
    factor_dict = compute_all_factors(
        prices, fundamentals=fundamentals, spy_prices=spy)
    calc_elapsed = time.time() - calc_start
    print(f"  计算耗时: {calc_elapsed:.1f} 秒")

    # 保存 + 处理
    print("\n[Step 4/4] 因子处理 + 存储...")
    for category, df in factor_dict.items():
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        processed = full_pipeline(df_clean, remove_redundant=False)
        save_factors(processed, category)
        print(f"  {category}: {len(processed):,} 行 x {len(processed.columns)} 因子")

    # 合并所有因子做冗余分析
    print("\n  冗余分析 (截面)...")
    all_factors = pd.concat(
        [df.replace([np.inf, -np.inf], np.nan) for df in factor_dict.values()
         if isinstance(df.index, pd.MultiIndex)],
        axis=1)
    if isinstance(all_factors.index, pd.MultiIndex) and len(all_factors) > 0:
        last_date = all_factors.index.get_level_values(0).max()
        cross_section = all_factors.loc[last_date].dropna(axis=1, how="all")
        corr = cross_section.corr()
        high_corr = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = abs(corr.iloc[i, j])
                if r > 0.8:
                    high_corr.append((cols[i], cols[j], round(r, 3)))
        if high_corr:
            print(f"  高相关因子对 (|r| > 0.8): {len(high_corr)} 对")
            for a, b, r in high_corr[:10]:
                print(f"    {a} ↔ {b}: r={r}")
        else:
            print("  无高相关冗余因子对")

    total_elapsed = time.time() - load_start
    total_factors = sum(len(df.columns) for df in factor_dict.values())
    print(f"\n{'=' * 60}")
    print(f"  因子计算完成! {total_factors} 个因子, 耗时 {total_elapsed:.1f} 秒")
    print(f"{'=' * 60}")

    cmd_stats(args)


def cmd_validate(args):
    """数据质量检查."""
    from factor_library.storage import load_prices
    from factor_library.validator import generate_quality_report

    print("[Validate] 加载价格数据...")
    prices = load_prices()
    if prices.empty:
        print("  无数据, 请先运行 --init")
        return
    generate_quality_report(prices)


def _load_factor_matrix(date: str = None):
    """辅助: 加载因子矩阵."""
    from factor_library.storage import load_factors
    from factor_library.search import build_factor_matrix

    categories = ["technical", "risk", "volatility", "liquidity", "fundamental"]
    factor_dfs = {}
    for cat in categories:
        df = load_factors(cat)
        if not df.empty:
            factor_dfs[cat] = df

    if not factor_dfs:
        print("  无因子数据, 请先运行 --compute-factors")
        return None

    return build_factor_matrix(factor_dfs, date=date)


def cmd_screen(args):
    """多因子选股."""
    print("=" * 60)
    print("  多因子选股引擎")
    print("=" * 60)

    matrix = _load_factor_matrix()
    if matrix is None:
        return

    print(f"\n  因子矩阵: {len(matrix)} 只 x {len(matrix.columns)} 因子")

    from factor_library.screener import (
        score_stocks, risk_filter, market_timing_signal,
        get_credit_spread_candidates, get_momentum_candidates,
    )

    # 市场状态
    timing = market_timing_signal(matrix)
    print(f"\n  市场状态: {timing['market_state']} (score={timing['score']})")
    for k, v in timing["signals"].items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
    print(f"  建议: {timing['recommendation']}")

    # 选股
    model = args.model or "momentum"
    top_n = args.top_n or 20
    print(f"\n  模型: {model} | Top {top_n}")

    if args.risk_filter:
        matrix = risk_filter(matrix)

    results = score_stocks(matrix, model=model, top_n=top_n)
    if results.empty:
        print("  无结果")
        return

    print(f"\n  {'Rank':>4} | {'Symbol':6} | {'Score':>8}")
    print(f"  {'-'*4} | {'-'*6} | {'-'*8}")
    for _, row in results.iterrows():
        print(f"  {row['rank']:4d} | {row['symbol']:6s} | {row['score']:8.4f}")


def cmd_similar(args):
    """相似股票搜索."""
    symbol = args.symbol.upper()
    print(f"\n  找和 {symbol} 最相似的股票...")

    matrix = _load_factor_matrix()
    if matrix is None:
        return

    from factor_library.search import find_similar
    results = find_similar(matrix, symbol, top_n=args.top_n or 10)
    if results.empty:
        print(f"  {symbol} 不在因子库中")
        return

    print(f"\n  {'Rank':>4} | {'Symbol':6} | {'Similarity':>10}")
    print(f"  {'-'*4} | {'-'*6} | {'-'*10}")
    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"  {i:4d} | {row['symbol']:6s} | {row['similarity']:10.4f}")


def cmd_report(args):
    """生成每日因子报告."""
    matrix = _load_factor_matrix()
    if matrix is None:
        return

    from factor_library.screener import generate_daily_report
    report = generate_daily_report(matrix)
    print(report)

    # 保存到文件
    from pathlib import Path
    report_dir = Path("docs/factor_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = report_dir / f"daily_{date_str}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  报告已保存: {report_path}")


def cmd_backtest(args):
    """因子回测验证: IC/IR + 五分位 + 模型回测."""
    from factor_library.storage import load_prices, load_factors
    from factor_library.backtest import generate_backtest_report

    print("=" * 60)
    print("  因子回测验证")
    print("=" * 60)

    print("\n[Step 1/2] 加载数据...")
    prices = load_prices()
    if prices.empty:
        print("  无价格数据, 请先运行 --init")
        return

    categories = ["technical", "risk", "volatility", "liquidity", "fundamental"]
    factor_dfs = {}
    for cat in categories:
        df = load_factors(cat)
        if not df.empty:
            factor_dfs[cat] = df
            print(f"  {cat}: {len(df.columns)} 因子, {len(df):,} 行")

    if not factor_dfs:
        print("  无因子数据, 请先运行 --compute-factors")
        return

    holding = getattr(args, "holding_days", 5)
    print(f"\n[Step 2/2] 运行回测 (holding={holding}d)...")
    start = time.time()
    report = generate_backtest_report(factor_dfs, prices, holding_days=holding)
    elapsed = time.time() - start
    print(report)
    print(f"\n  回测耗时: {elapsed:.1f} 秒")

    from pathlib import Path
    report_dir = Path("docs/factor_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = report_dir / f"backtest_{date_str}.txt"
    path.write_text(report, encoding="utf-8")
    print(f"  报告已保存: {path}")


def cmd_anomalies(args):
    """异常股票检测."""
    matrix = _load_factor_matrix()
    if matrix is None:
        return

    from factor_library.search import find_anomalies
    results = find_anomalies(matrix, top_n=args.top_n or 20)
    print(f"\n  因子异常 Top {len(results)}:")
    print(f"  {'Rank':>4} | {'Symbol':6} | {'Score':>7} | {'Extreme Factors'}")
    print(f"  {'-'*4} | {'-'*6} | {'-'*7} | {'-'*30}")
    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"  {i:4d} | {row['symbol']:6s} | {row['anomaly_score']:7.2f} | "
              f"{row['extreme_factors']}")


def main():
    parser = argparse.ArgumentParser(description="美股因子库管理")
    sub = parser.add_subparsers(dest="command")

    # 原有命令 (保持兼容)
    parser.add_argument("--init", action="store_true", help="首次初始化")
    parser.add_argument("--update", action="store_true", help="每日增量更新")
    parser.add_argument("--fundamentals", action="store_true", help="更新基本面")
    parser.add_argument("--compute-factors", action="store_true", help="全量计算因子")
    parser.add_argument("--stats", action="store_true", help="查看状态")
    parser.add_argument("--validate", action="store_true", help="数据质量检查")

    # 新增 Phase 3/4 命令
    parser.add_argument("--screen", action="store_true",
                        help="多因子选股 (搭配 --model)")
    parser.add_argument("--similar", type=str, default="",
                        help="找相似股票 (e.g. --similar AAPL)")
    parser.add_argument("--report", action="store_true",
                        help="生成每日因子报告")
    parser.add_argument("--anomalies", action="store_true",
                        help="检测因子异常股票")
    parser.add_argument("--backtest", action="store_true",
                        help="因子回测验证 (IC/IR + 五分位 + 模型)")

    # 参数
    parser.add_argument("--model", type=str, default="momentum",
                        help="选股模型 (value/momentum/quality/low_risk/credit_spread)")
    parser.add_argument("--top-n", type=int, default=20, help="返回数量")
    parser.add_argument("--risk-filter", action="store_true", help="启用风控过滤")
    parser.add_argument("--limit", type=int, default=0, help="限制标的数量")
    parser.add_argument("--years", type=int, default=5, help="历史年数")
    parser.add_argument("--fresh", action="store_true", help="忽略断点续传")
    parser.add_argument("--skip-fundamentals", action="store_true",
                        help="跳过基本面下载")
    parser.add_argument("--skip-sectors", action="store_true",
                        help="跳过行业分类")
    parser.add_argument("--holding-days", type=int, default=5,
                        help="回测前瞻天数 (默认 5)")
    parser.add_argument("--train-regime", action="store_true",
                        help="训练 HMM 市场状态模型")
    parser.add_argument("--regime", action="store_true",
                        help="检测当前市场状态 (HMM)")

    args = parser.parse_args()

    # 处理 --similar 的特殊逻辑
    if args.similar:
        args.symbol = args.similar
        cmd_similar(args)
    elif args.init:
        cmd_init(args)
    elif args.update:
        cmd_update(args)
    elif args.fundamentals:
        cmd_fundamentals(args)
    elif getattr(args, "compute_factors", False):
        cmd_compute_factors(args)
    elif args.screen:
        cmd_screen(args)
    elif args.report:
        cmd_report(args)
    elif args.anomalies:
        cmd_anomalies(args)
    elif getattr(args, "backtest", False):
        cmd_backtest(args)
    elif args.validate:
        cmd_validate(args)
    elif getattr(args, "train_regime", False):
        from factor_library.regime import train_regime_model
        print("[HMM] Training market regime model...")
        result = train_regime_model(years=args.years)
        print(f"  Current regime: {result['current']}")
        for name, s in result["regimes"].items():
            print(f"  {name:12s}: {s['pct']:5.1f}% of days, "
                  f"ann.ret={s['avg_return']:+6.1f}%, vol={s['avg_vol']:.1f}%")
    elif getattr(args, "regime", False):
        from factor_library.regime import get_current_regime
        r = get_current_regime()
        print(f"  Regime:     {r['regime']}")
        print(f"  State:      {r['market_state']}")
        print(f"  Confidence: {r['confidence']:.1%}")
        if r.get("recent_distribution"):
            print(f"  Recent 20d: {r['recent_distribution']}")
    elif args.stats:
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
