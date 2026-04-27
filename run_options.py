"""
FUTU-QUANT 期权策略交易系统 - 主入口

Usage:
    python run_options.py --strategy orb --dry-run      # 0DTE ORB 干跑
    python run_options.py --strategy all --dry-run       # 全部策略干跑
    python run_options.py --strategy credit_spread       # Credit Spread 实盘
    python run_options.py --backtest --strategy orb       # 回测
"""
from __future__ import annotations

import sys
import io
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Load .env if present
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from utils.helpers import load_yaml, get_project_root
from utils.logger import setup_logger
from data.trade_store import TradeStore
from options.order import OptionTrade
from options.risk import OptionsRiskManager

logger = setup_logger("run_options")

STRATEGY_NAMES = ["orb", "credit_spread", "earnings", "straddle", "wheel", "pmcc"]


def load_options_config() -> dict:
    cfg_path = get_project_root() / "config" / "options.yaml"
    cfg = load_yaml(str(cfg_path))
    return cfg.get("options", {})


def run_live(strategies: list[str], dry_run: bool, config: dict):
    """Run live/simulate option strategies."""
    from futu import OpenQuoteContext, OpenSecTradeContext, TrdEnv

    trade_env = config.get("trade_env", "SIMULATE")
    logger.info(f"启动期权交易系统 | 环境={trade_env} | dry_run={dry_run} | 策略={strategies}")

    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    trade_ctx = OpenSecTradeContext(host="127.0.0.1", port=11111)
    store = TradeStore()

    from options.chain import OptionChainFetcher
    from options.trader import OptionsTrader
    from options.scanner import VolatilityScanner

    from risk.pdt_guard import PdtGuard

    chain = OptionChainFetcher(quote_ctx)
    pdt_guard = PdtGuard(max_day_trades=3, rolling_window_days=5, trade_store=store)
    risk_mgr = OptionsRiskManager(config, pdt_guard=pdt_guard)
    fmp_key = config.get("fmp_api_key", "") or os.environ.get("FMP_API_KEY", "")
    trader = OptionsTrader(trade_ctx, quote_ctx, risk_mgr,
                           trade_env=trade_env, dry_run=dry_run,
                           fmp_api_key=fmp_key)
    scanner = VolatilityScanner(quote_ctx)

    try:
        # ── Strategy A: 0DTE ORB ──
        if "orb" in strategies or "all" in strategies:
            _run_orb(chain, trader, store, config.get("orb", {}), dry_run, config)

        # ── Strategy B: Earnings Spread ──
        if "earnings" in strategies or "all" in strategies:
            _run_earnings(chain, trader, store, config.get("earnings", {}), dry_run)

        # ── Strategy C: Credit Spread ──
        if "credit_spread" in strategies or "all" in strategies:
            _run_credit_spread(chain, trader, scanner, store,
                               config.get("credit_spread", {}), config, dry_run)

        # ── Strategy D: Straddle Squeeze ──
        if "straddle" in strategies or "all" in strategies:
            straddle_cfg = config.get("straddle", {})
            if not straddle_cfg.get("enabled", True):
                logger.info("[STRADDLE] 策略已禁用，跳过")
            else:
                _run_straddle(chain, trader, scanner, store, straddle_cfg, dry_run)

        # ── Strategy E: Wheel (CSP → Covered Call) ──
        if "wheel" in strategies or "all" in strategies:
            wheel_cfg = config.get("wheel", {})
            if not wheel_cfg.get("enabled", True):
                logger.info("[WHEEL] 策略已禁用，跳过")
            else:
                _run_wheel(chain, trader, scanner, store, wheel_cfg, config, dry_run)

        # ── Strategy F: PMCC (Poor Man's Covered Call) ──
        if "pmcc" in strategies or "all" in strategies:
            pmcc_cfg = config.get("pmcc", {})
            if not pmcc_cfg.get("enabled", True):
                logger.info("[PMCC] 策略已禁用，跳过")
            else:
                _run_pmcc(chain, trader, scanner, store, pmcc_cfg, config, dry_run)

        # Monitor loop
        logger.info("进入持仓监控循环 (Ctrl+C 退出)...")
        while trader.get_open_trades():
            trader.monitor_open_trades()
            time.sleep(30)

    except KeyboardInterrupt:
        logger.info("用户中断，退出...")
    finally:
        _save_open_trades(trader, store, dry_run)
        quote_ctx.close()
        trade_ctx.close()
        store.close()
        logger.info("期权交易系统已关闭")


def _run_orb(chain, trader, store, config, dry_run, full_config=None):
    from options.strategies.orb_0dte import ORBStrategy
    from futu import RET_OK, KLType

    per_strat = {}
    if full_config:
        per_strat = full_config.get("risk", {}).get("per_strategy", {}).get("orb", {})
    max_loss = float(per_strat.get("max_loss_per_trade", 200))

    strat = ORBStrategy(config, chain, max_loss_per_trade=max_loss)
    strat.reset_daily()

    for sym in strat.symbols:
        logger.info(f"[ORB] 获取 {sym} 日内数据...")
        time.sleep(0.5)
        ret, data, _ = chain._ctx.request_history_kline(
            sym, ktype=KLType.K_5M, max_count=78
        )
        if ret != RET_OK or data is None or len(data) < 12:
            logger.warning(f"[ORB] {sym} 日内数据不足")
            continue

        # Update ORB range from first 6 bars
        for i, (_, bar) in enumerate(data.head(6).iterrows()):
            bar_time = datetime.strptime(bar["time_key"], "%Y-%m-%d %H:%M:%S")
            strat.update_orb(sym, bar["high"], bar["low"], bar_time)

        # Mark ORB as ready
        strat._orb_ready[sym] = True

        # Check for breakout
        latest = data.iloc[-1]
        trade = strat.evaluate(sym, latest["close"])
        if trade:
            trade.dry_run = dry_run
            if trader.open_trade(trade):
                trade.trade_id = store.log_option_trade(
                    strategy=trade.strategy, underlying=trade.underlying,
                    legs_json=trade.legs_json(), max_loss=trade.max_loss,
                    target_pnl=trade.target_pnl, dry_run=dry_run,
                )


def _run_earnings(chain, trader, store, config, dry_run):
    from options.strategies.earnings_spread import EarningsSpreadStrategy

    strat = EarningsSpreadStrategy(config, chain)
    # Simplified: without real earnings calendar, log what would happen
    logger.info("[EARNINGS] 财报价差策略需要财报日历数据源。当前跳过扫描。")
    logger.info("[EARNINGS] 可通过 --strategy earnings 配合手动传入财报日期运行。")


def _run_credit_spread(chain, trader, scanner, store, config, full_config, dry_run):
    from options.strategies.credit_spread import CreditSpreadStrategy

    symbols = config.get("scan_symbols", [])
    min_ivr = config.get("min_ivr", 60)
    max_pos = config.get("max_positions", 3)

    per_strat = full_config.get("risk", {}).get("per_strategy", {}).get("credit_spread", {})
    min_credit = float(per_strat.get("min_credit_received", 0.15))

    logger.info(f"[CREDIT] 扫描 {len(symbols)} 个标的 (min IVR={min_ivr}, min credit=${min_credit:.2f})...")
    ivr_hits = scanner.scan_ivr(symbols, min_ivr=min_ivr)

    strat = CreditSpreadStrategy(config, chain, min_credit=min_credit)
    opened = 0
    for hit in ivr_hits:
        if opened >= max_pos:
            break
        trade = strat.evaluate(hit["symbol"], hit["ivr"], min_ivr)
        if trade:
            trade.dry_run = dry_run
            if trader.open_trade(trade):
                trade.trade_id = store.log_option_trade(
                    strategy=trade.strategy, underlying=trade.underlying,
                    legs_json=trade.legs_json(), max_loss=trade.max_loss,
                    target_pnl=trade.target_pnl, dry_run=dry_run,
                )
                opened += 1

    logger.info(f"[CREDIT] 开仓 {opened} 组 Credit Spread")


def _run_straddle(chain, trader, scanner, store, config, dry_run):
    from options.strategies.straddle_squeeze import StraddleSqueezeStrategy

    symbols = config.get("symbols", ["US.GLD", "US.TLT"])
    lookback = config.get("bb_width_lookback", 126)
    percentile = config.get("bb_width_percentile", 5)

    logger.info(f"[STRADDLE] 扫描 {len(symbols)} 个标的 BB 挤压...")
    squeezes = scanner.scan_bb_squeeze(symbols, lookback=lookback, percentile=percentile)

    strat = StraddleSqueezeStrategy(config, chain)
    for sq in squeezes:
        trade = strat.evaluate(sq["symbol"], sq)
        if trade:
            trade.dry_run = dry_run
            if trader.open_trade(trade):
                trade.trade_id = store.log_option_trade(
                    strategy=trade.strategy, underlying=trade.underlying,
                    legs_json=trade.legs_json(), max_loss=trade.max_loss,
                    target_pnl=trade.target_pnl, dry_run=dry_run,
                )


def _run_wheel(chain, trader, scanner, store, config, full_config, dry_run):
    from options.strategies.wheel import WheelStrategy
    from futu import RET_OK

    symbols = config.get("symbols", [])
    min_ivr = config.get("min_ivr", 30)

    per_strat = full_config.get("risk", {}).get("per_strategy", {}).get("wheel", {})
    max_pos = int(per_strat.get("max_open_positions", 1))

    logger.info(f"[WHEEL] 扫描 {len(symbols)} 个标的 (max_price=${config.get('max_stock_price', 30)}, "
                f"min_ivr={min_ivr})...")

    # Get current prices and IVR for each symbol
    strat = WheelStrategy(config, chain)
    opened = 0
    for sym in symbols:
        if opened >= max_pos:
            break

        # Get current price
        import time; time.sleep(0.5)
        ret, snap = chain._ctx.get_market_snapshot([sym])
        if ret != RET_OK or snap is None or len(snap) == 0:
            continue
        price = float(snap.iloc[0]["last_price"])

        if price > config.get("max_stock_price", 30):
            logger.info(f"[WHEEL] {sym} ${price:.2f} > max ${config.get('max_stock_price', 30)}, skip")
            continue

        # Simple IVR using scanner data
        ivr_hits = scanner.scan_ivr([sym], min_ivr=min_ivr)
        if not ivr_hits:
            continue
        ivr = ivr_hits[0]["ivr"]

        trade = strat.evaluate_csp(sym, ivr, price)
        if trade:
            trade.dry_run = dry_run
            if trader.open_trade(trade):
                trade.trade_id = store.log_option_trade(
                    strategy=trade.strategy, underlying=trade.underlying,
                    legs_json=trade.legs_json(), max_loss=trade.max_loss,
                    target_pnl=trade.target_pnl, dry_run=dry_run,
                )
                opened += 1

    logger.info(f"[WHEEL] 开仓 {opened} 组 CSP")


def _run_pmcc(chain, trader, scanner, store, config, full_config, dry_run):
    from options.strategies.pmcc import PMCCStrategy

    symbols = config.get("symbols", [])
    logger.info(f"[PMCC] 扫描 {len(symbols)} 个标的...")

    strat = PMCCStrategy(config, chain)
    for sym in symbols:
        trade = strat.evaluate_leaps(sym)
        if trade:
            trade.dry_run = dry_run
            if trader.open_trade(trade):
                trade.trade_id = store.log_option_trade(
                    strategy=trade.strategy, underlying=trade.underlying,
                    legs_json=trade.legs_json(), max_loss=trade.max_loss,
                    target_pnl=trade.target_pnl, dry_run=dry_run,
                )


def _save_open_trades(trader, store, dry_run):
    for trade in trader.get_open_trades():
        if trade.trade_id and trade.status == "closed":
            store.close_option_trade(trade.trade_id, trade.realized_pnl, trade.close_reason)


def run_backtest(strategies: list[str], config: dict):
    """Run synthetic options backtest."""
    from futu import OpenQuoteContext, RET_OK, KLType
    from options.backtest import OptionsBacktester, print_backtest_report

    logger.info(f"启动期权回测 | 策略={strategies}")
    bt = OptionsBacktester()

    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)

    try:
        if "orb" in strategies or "all" in strategies:
            for sym in config.get("orb", {}).get("symbols", ["US.SPY"]):
                logger.info(f"[BT] ORB 回测: {sym} (5分钟数据)")
                time.sleep(0.5)
                ret, data, _ = quote_ctx.request_history_kline(
                    sym, ktype=KLType.K_5M, max_count=1000
                )
                if ret == RET_OK and data is not None and len(data) > 50:
                    result = bt.backtest_orb(data, config.get("orb", {}))
                    print_backtest_report(result)
                else:
                    logger.warning(f"[BT] {sym} 5分钟数据不足")

        if "credit_spread" in strategies or "all" in strategies:
            for sym in config.get("credit_spread", {}).get("scan_symbols", ["US.SPY"])[:3]:
                logger.info(f"[BT] Credit Spread 回测: {sym}")
                time.sleep(0.5)
                ret, data, _ = quote_ctx.request_history_kline(
                    sym, ktype=KLType.K_DAY, max_count=500
                )
                if ret == RET_OK and data is not None and len(data) > 100:
                    result = bt.backtest_credit_spread(data, config.get("credit_spread", {}))
                    print_backtest_report(result)

        if "straddle" in strategies or "all" in strategies:
            for sym in config.get("straddle", {}).get("symbols", ["US.GLD"])[:2]:
                logger.info(f"[BT] Straddle 回测: {sym}")
                time.sleep(0.5)
                ret, data, _ = quote_ctx.request_history_kline(
                    sym, ktype=KLType.K_DAY, max_count=500
                )
                if ret == RET_OK and data is not None and len(data) > 200:
                    result = bt.backtest_straddle(data, config.get("straddle", {}))
                    print_backtest_report(result)

    finally:
        quote_ctx.close()


def show_status(config: dict):
    """Show current option positions and risk status."""
    store = TradeStore()

    print(f"\n{'=' * 60}")
    print("  期权交易系统状态")
    print(f"{'=' * 60}")

    open_trades = store.query_option_trades(status="open")
    print(f"\n  持仓中: {len(open_trades)} 笔")
    for t in open_trades:
        print(f"    [{t['strategy']}] {t['underlying']} | "
              f"max_loss=${t['max_loss']:.0f} | 开仓: {t['timestamp'][:16]}")

    recent = store.query_option_trades(days=7)
    closed = [t for t in recent if t["status"] == "closed"]
    if closed:
        total_pnl = sum(t["realized_pnl"] for t in closed)
        wins = sum(1 for t in closed if t["realized_pnl"] > 0)
        print(f"\n  近 7 天已平仓: {len(closed)} 笔")
        print(f"  总盈亏: ${total_pnl:,.2f}")
        print(f"  胜率: {wins / len(closed) * 100:.0f}%")

    risk = OptionsRiskManager(config)
    status = risk.get_status()
    print(f"\n  资金配置:")
    print(f"    总资金:       ${status['capital']:,.0f}")
    print(f"    单日亏损限额: ${status['daily_limit']:,.0f}")
    print(f"    月度回撤限额: ${status['monthly_limit']:,.0f}")
    print(f"{'=' * 60}\n")

    store.close()


def main():
    parser = argparse.ArgumentParser(description="期权策略交易系统")
    parser.add_argument("--strategy", type=str, default="all",
                        help="策略: orb, credit_spread, earnings, straddle, all")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑模式 (不下单)")
    parser.add_argument("--backtest", action="store_true",
                        help="运行回测")
    parser.add_argument("--status", action="store_true",
                        help="查看持仓和风控状态")
    args = parser.parse_args()

    config = load_options_config()
    if not config.get("enabled", False):
        print("期权交易系统未启用。请在 config/options.yaml 中设置 enabled: true")
        return

    strategies = [args.strategy] if args.strategy != "all" else STRATEGY_NAMES

    if args.status:
        show_status(config)
    elif args.backtest:
        run_backtest(strategies, config)
    else:
        run_live(strategies, args.dry_run, config)


if __name__ == "__main__":
    main()
