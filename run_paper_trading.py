"""
Paper Trading CLI — 因子模型模拟盘验证

Usage:
    python run_paper_trading.py --init            # 初始化组合 (首次)
    python run_paper_trading.py --daily           # 每日 NAV 快照
    python run_paper_trading.py --rebalance       # 月度调仓
    python run_paper_trading.py --report          # 查看报告
    python run_paper_trading.py --status          # 当前状态
    python run_paper_trading.py --futu-rebalance  # Futu 模拟环境同步调仓
    python run_paper_trading.py --compare         # Paper vs Futu 滑点对比
    python run_paper_trading.py --telegram        # 发送 Telegram 周报
"""
from __future__ import annotations

import sys
import io
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import os
from datetime import datetime, date
from pathlib import Path

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from utils.logger import setup_logger
from backtest.paper_trading import PaperTrader, MODELS

logger = setup_logger("run_paper_trading")


def cmd_init(args):
    pt = PaperTrader(capital=args.capital)
    pt.init_portfolios()
    print(f"Initialized {len(MODELS)} portfolios @ ${args.capital:,.0f} each")

    if args.rebalance_now:
        pt.rebalance_all()
        pt.record_daily_nav()

    pt.close()


def cmd_daily(args):
    pt = PaperTrader()
    pt.record_daily_nav()

    # Auto-rebalance on 1st of month
    today = date.today()
    if today.day == 1 or args.force_rebalance:
        logger.info("Monthly rebalance triggered")
        pt.rebalance_all()
        pt.record_daily_nav()

    pt.close()


def cmd_rebalance(args):
    pt = PaperTrader()
    if args.model:
        pt.rebalance(args.model)
    else:
        pt.rebalance_all()
    pt.record_daily_nav()
    pt.close()


def cmd_report(args):
    pt = PaperTrader()
    report = pt.generate_report()
    print(report)
    pt.close()


def cmd_status(args):
    pt = PaperTrader()

    print(f"\n{'=' * 50}")
    print("  Paper Trading Status")
    print(f"{'=' * 50}")

    for model in MODELS:
        holdings = pt.get_holdings(model)
        cash = pt.get_cash(model)
        nav_df = pt.get_nav_series(model)

        if not nav_df.empty:
            nav = nav_df["nav"].iloc[-1]
            ret = (nav / pt.initial_capital - 1) * 100
            print(f"\n  {model.upper()}: NAV=${nav:,.2f} ({ret:+.2f}%)")
        else:
            print(f"\n  {model.upper()}: NAV=${pt.initial_capital:,.2f} (no data)")

        print(f"    Holdings: {len(holdings)} | Cash: ${cash:,.2f}")

        if holdings:
            for h in sorted(holdings, key=lambda x: x["symbol"]):
                print(f"      {h['symbol']:6s}  {h['shares']:8.2f} sh "
                      f"@ ${h['cost_price']:.2f}")

    print(f"\n{'=' * 50}\n")
    pt.close()


def cmd_futu_rebalance(args):
    from backtest.futu_paper import FutuPaperTrader

    pt = PaperTrader()
    ft = FutuPaperTrader()

    if not ft.connect():
        print("Cannot connect to Futu OpenD")
        return

    try:
        model = args.model or "momentum"
        holdings = pt.get_holdings(model)
        target_symbols = [h["symbol"] for h in holdings]

        if not target_symbols:
            print(f"No holdings in {model}, run --rebalance first")
            return

        print(f"Executing Futu SIMULATE rebalance for {model}")
        print(f"Target: {len(target_symbols)} stocks")

        summary = ft.execute_rebalance(target_symbols)
        print(f"Done: {summary['sells']} sells, {summary['buys']} buys")
    finally:
        ft.disconnect()
        pt.close()


def cmd_compare(args):
    from backtest.futu_paper import FutuPaperTrader

    pt = PaperTrader()
    ft = FutuPaperTrader()

    if not ft.connect():
        print("Cannot connect to Futu OpenD")
        return

    try:
        model = args.model or "momentum"
        paper_holdings = pt.get_holdings(model)

        if not paper_holdings:
            print(f"No holdings in {model}")
            return

        result = ft.compare_with_paper(paper_holdings)
        print(f"\nPaper vs Futu Slippage ({model})")
        print(f"  Compared: {result['n_compared']} positions")
        print(f"  Avg Slippage: {result['avg_slippage_bps']:.2f} bps")

        if result["details"]:
            print(f"\n  {'Symbol':8s} {'Paper':>8s} {'Futu':>8s} {'Slip(bps)':>10s}")
            for d in result["details"]:
                print(f"  {d['symbol']:8s} "
                      f"${d['paper_cost']:7.2f} "
                      f"${d['futu_cost']:7.2f} "
                      f"{d['slippage_bps']:9.2f}")
    finally:
        ft.disconnect()
        pt.close()


def cmd_telegram(args):
    from notification.telegram_bot import TelegramNotifier

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not tg_token or not tg_chat:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
        return

    pt = PaperTrader()
    report = pt.generate_telegram_report()
    pt.close()

    notifier = TelegramNotifier(tg_token, tg_chat)
    notifier.send_sync(report)
    print("Telegram report sent")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading CLI")
    sub = parser.add_subparsers(dest="command")

    # Flat flags for simplicity
    parser.add_argument("--init", action="store_true",
                        help="Initialize portfolios")
    parser.add_argument("--daily", action="store_true",
                        help="Daily NAV snapshot")
    parser.add_argument("--rebalance", action="store_true",
                        help="Run monthly rebalance")
    parser.add_argument("--report", action="store_true",
                        help="Print performance report")
    parser.add_argument("--status", action="store_true",
                        help="Show current portfolio status")
    parser.add_argument("--futu-rebalance", action="store_true",
                        help="Execute rebalance in Futu SIMULATE")
    parser.add_argument("--compare", action="store_true",
                        help="Compare Paper vs Futu slippage")
    parser.add_argument("--telegram", action="store_true",
                        help="Send Telegram weekly report")
    parser.add_argument("--model", type=str, default="",
                        help="Specific model (value/low_risk/momentum)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital per model (default: 10000)")
    parser.add_argument("--rebalance-now", action="store_true",
                        help="Immediately rebalance after init")
    parser.add_argument("--force-rebalance", action="store_true",
                        help="Force rebalance on --daily regardless of date")

    args = parser.parse_args()

    if args.init:
        cmd_init(args)
    elif args.daily:
        cmd_daily(args)
    elif args.rebalance:
        cmd_rebalance(args)
    elif args.report:
        cmd_report(args)
    elif args.status:
        cmd_status(args)
    elif args.futu_rebalance:
        cmd_futu_rebalance(args)
    elif args.compare:
        cmd_compare(args)
    elif args.telegram:
        cmd_telegram(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
