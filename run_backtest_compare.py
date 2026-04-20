"""
FUTU-QUANT 策略参数对比回测
对比默认参数 vs 激进参数在不同标的上的表现
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from futu import OpenQuoteContext, RET_OK, KLType, SubType
from backtest.backtester import Backtester
from backtest.report import BacktestReport
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.breakout import BreakoutStrategy
from utils.helpers import load_yaml


def fetch_history(symbol: str, days: int = 200) -> pd.DataFrame:
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    ctx.subscribe([symbol], [SubType.K_DAY])
    ret, data = ctx.get_cur_kline(symbol, days, KLType.K_DAY)
    ctx.close()
    if ret != RET_OK:
        return pd.DataFrame()
    return data


def run_backtest(strategy, symbol, data, capital=3000):
    bt = Backtester(initial_capital=capital, commission_pct=0.001, slippage_pct=0.0005)
    result = bt.run(strategy, symbol, data)
    report = BacktestReport(result)
    return report.summary()


def main():
    default_cfg = load_yaml("config/strategies.yaml")
    aggressive_cfg = load_yaml("config/strategies_aggressive.yaml")

    symbols = ["US.TQQQ", "US.SQQQ", "US.SOXL", "US.SPY", "US.QQQ"]

    param_sets = {
        "默认参数": default_cfg,
        "激进参数": aggressive_cfg,
    }

    strategy_builders = {
        "momentum": lambda p: MomentumStrategy(params=p),
        "mean_reversion": lambda p: MeanReversionStrategy(params=p),
        "breakout": lambda p: BreakoutStrategy(params=p),
    }

    strategy_names = {
        "momentum": "动量",
        "mean_reversion": "均值回归",
        "breakout": "突破",
    }

    print()
    print("=" * 70)
    print("  FUTU-QUANT 策略参数对比回测")
    print("  默认参数 vs 激进参数")
    print("=" * 70)

    # 先拉取所有数据
    all_data = {}
    for symbol in symbols:
        data = fetch_history(symbol, 200)
        if not data.empty:
            all_data[symbol] = data
            print(f"  [OK] {symbol}: {len(data)} 条日线 ({str(data['time_key'].iloc[0])[:10]} ~ {str(data['time_key'].iloc[-1])[:10]})")
        else:
            print(f"  [FAIL] {symbol}: 数据获取失败")

    results = []

    for strat_key, strat_name in strategy_names.items():
        for param_label, cfg in param_sets.items():
            params = cfg["strategies"][strat_key]["params"]
            for symbol, data in all_data.items():
                strategy = strategy_builders[strat_key](params)
                summary = run_backtest(strategy, symbol, data)
                results.append({
                    "策略": strat_name,
                    "参数": param_label,
                    "标的": symbol,
                    "收益率%": summary["total_return_pct"],
                    "最大回撤%": summary["max_drawdown_pct"],
                    "交易次数": summary["total_trades"],
                    "胜率%": summary["win_rate_pct"],
                    "盈亏比": summary["profit_factor"],
                })

    # 打印对比表
    print(f"\n\n{'=' * 90}")
    print("  对比结果")
    print(f"{'=' * 90}")

    print(f"\n  {'策略':<8} {'参数':<8} {'标的':<12} {'收益率':>10} {'回撤':>8} {'交易数':>6} {'胜率':>8} {'盈亏比':>8}")
    print("  " + "-" * 84)

    current_strat = ""
    for r in results:
        if r["策略"] != current_strat:
            current_strat = r["策略"]
            if r != results[0]:
                print()

        pf = f"{r['盈亏比']:.2f}" if r['盈亏比'] != float('inf') else "inf"
        print(f"  {r['策略']:<8} {r['参数']:<8} {r['标的']:<12} {r['收益率%']:>+8.2f}% {r['最大回撤%']:>7.2f}% {r['交易次数']:>5} {r['胜率%']:>7.1f}% {pf:>8}")

    # 最佳组合
    print(f"\n\n{'=' * 70}")
    print("  最佳策略组合 TOP 5 (按收益率排序)")
    print(f"{'=' * 70}\n")

    sorted_results = sorted(results, key=lambda x: x["收益率%"], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        if r["交易次数"] > 0 or r["收益率%"] != 0:
            pf = f"{r['盈亏比']:.2f}" if r['盈亏比'] != float('inf') else "inf"
            print(f"  #{i+1}  {r['策略']} ({r['参数']}) @ {r['标的']}")
            print(f"      收益: {r['收益率%']:+.2f}% | 回撤: {r['最大回撤%']:.2f}% | 交易: {r['交易次数']}次 | 胜率: {r['胜率%']:.1f}% | 盈亏比: {pf}")
            print()

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
