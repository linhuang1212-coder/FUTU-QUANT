"""
FUTU-QUANT 策略回测脚本
使用真实历史数据回测三个内置策略
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


def fetch_history(symbol: str, days: int = 200) -> pd.DataFrame:
    """从富途拉取历史日线数据"""
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    ret_sub, _ = ctx.subscribe([symbol], [SubType.K_DAY])

    ret, data = ctx.get_cur_kline(symbol, days, KLType.K_DAY)
    ctx.close()

    if ret != RET_OK:
        print(f"[FAIL] 获取 {symbol} 历史数据失败: {data}")
        return pd.DataFrame()

    data = data.rename(columns={
        "time_key": "time_key",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    })
    return data


def run_single_backtest(strategy, symbol: str, data: pd.DataFrame, capital: float = 3000):
    """运行单个策略回测"""
    bt = Backtester(initial_capital=capital, commission_pct=0.001, slippage_pct=0.0005)
    result = bt.run(strategy, symbol, data)
    report = BacktestReport(result)
    return report.summary(), result


def print_summary(name: str, summary: dict):
    """打印回测摘要"""
    print(f"\n  {'=' * 55}")
    print(f"  策略: {name}")
    print(f"  {'=' * 55}")
    print(f"  初始资金:     ${summary['initial_capital']:>10,.2f}")
    print(f"  最终资金:     ${summary['final_capital']:>10,.2f}")
    print(f"  总收益:       ${summary['total_return']:>+10,.2f} ({summary['total_return_pct']:>+.2f}%)")
    print(f"  最大回撤:      {summary['max_drawdown_pct']:>10.2f}%")
    print(f"  交易次数:      {summary['total_trades']:>10}")
    print(f"  胜率:          {summary['win_rate_pct']:>10.2f}%")
    print(f"  平均盈利:     ${summary['avg_win']:>10,.2f}")
    print(f"  平均亏损:     ${summary['avg_loss']:>10,.2f}")
    print(f"  盈亏比:        {summary['profit_factor']:>10.2f}")
    print(f"  总手续费:     ${summary['total_commission']:>10,.2f}")
    print(f"  {'=' * 55}")


def main():
    symbols = ["US.TQQQ", "US.SQQQ", "US.SOXL", "US.SPY", "US.QQQ"]

    strategies = {
        "Momentum (动量)": MomentumStrategy(),
        "MeanReversion (均值回归)": MeanReversionStrategy(),
        "Breakout (突破)": BreakoutStrategy(),
    }

    print()
    print("=" * 60)
    print("  FUTU-QUANT 策略回测")
    print("  回测周期: 近 200 个交易日")
    print("  初始资金: $3,000")
    print("=" * 60)

    all_results = {}

    for symbol in symbols:
        print(f"\n\n{'#' * 60}")
        print(f"  标的: {symbol}")
        print(f"{'#' * 60}")

        data = fetch_history(symbol, 200)
        if data.empty:
            print(f"  跳过 {symbol}，无法获取数据")
            continue

        print(f"  数据范围: {str(data['time_key'].iloc[0])[:10]} ~ {str(data['time_key'].iloc[-1])[:10]}")
        print(f"  数据条数: {len(data)}")
        print(f"  价格范围: ${data['close'].min():.2f} ~ ${data['close'].max():.2f}")

        for name, strategy in strategies.items():
            strategy_copy = type(strategy)(params=strategy.get_params())
            summary, result = run_single_backtest(strategy_copy, symbol, data)
            print_summary(f"{name} @ {symbol}", summary)
            all_results[f"{name} @ {symbol}"] = summary

    # 总结对比
    print(f"\n\n{'=' * 60}")
    print("  策略回测对比汇总")
    print(f"{'=' * 60}")
    print(f"\n  {'策略+标的':<40} {'收益率':>10} {'最大回撤':>10} {'胜率':>8} {'交易次数':>8}")
    print("  " + "-" * 78)

    for key, s in all_results.items():
        ret_str = f"{s['total_return_pct']:+.2f}%"
        dd_str = f"{s['max_drawdown_pct']:.2f}%"
        wr_str = f"{s['win_rate_pct']:.1f}%"
        trades_str = f"{s['total_trades']}"
        print(f"  {key:<40} {ret_str:>10} {dd_str:>10} {wr_str:>8} {trades_str:>8}")

    print()
    print("=" * 60)
    print("  回测完成!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
