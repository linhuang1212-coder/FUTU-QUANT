"""
FUTU-QUANT 一键启动 — 全自动期权 + ETF 交易

启动后自动执行:
  1. Cash Parking: 闲置资金自动买入 SGOV (短期国债 ETF)
  2. 动量轮动: 月度 ETF 轮动 (12M-1M 动量 Top2)
  3. 扫描 Credit Spread 机会 (IVR>60, 先买后卖绕过保证金限制)
  4. 检查 ORB 日内信号 (仅 SPY/QQQ, PDT安全)
  5. 持仓监控 (止盈/止损自动平仓)
  6. Telegram 实时推送

Usage:
  python start.py              # 实盘交易 (REAL)
  python start.py --dry-run    # 干跑模式 (不下单)
  python start.py --status     # 查看当前持仓
"""
import sys
import os

# Ensure UTF-8 output on Windows
import io
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load .env
from pathlib import Path
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _acquire_lock():
    """Prevent multiple instances from running simultaneously."""
    lock_path = Path(__file__).resolve().parent / ".start.lock"
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            import psutil
            if psutil.pid_exists(pid):
                print(f"另一个实例正在运行 (PID={pid})。")
                print(f"如确认无其他实例，删除 {lock_path} 后重试。")
                sys.exit(1)
        except (ImportError, ValueError):
            pass
    lock_path.write_text(str(os.getpid()))
    return lock_path


def _release_lock(lock_path):
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass

def is_us_market_open() -> bool:
    """Check if US stock market is currently open (or within 5 min of open)."""
    from datetime import datetime, timezone, timedelta
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        et = timezone(timedelta(hours=-4))
    now_et = datetime.now(et)
    weekday = now_et.weekday()
    if weekday >= 5:
        return False
    hour, minute = now_et.hour, now_et.minute
    t = hour * 60 + minute
    # Regular session: 9:30 - 16:00 ET (allow 5 min early for pre-open)
    return 9 * 60 + 25 <= t <= 16 * 60


def load_fractional_config() -> dict:
    """Load fractional strategy config."""
    from utils.helpers import load_yaml
    cfg_path = Path(__file__).resolve().parent / "config" / "fractional.yaml"
    if cfg_path.exists():
        cfg = load_yaml(str(cfg_path))
        return cfg.get("fractional", {})
    return {}


def run_etf_strategies(dry_run: bool = False, trade_env: str = "REAL"):
    """Run Cash Parking + Momentum Rotation ETF strategies."""
    from futu import OpenQuoteContext, OpenSecTradeContext, TrdEnv
    from strategy.fractional.cash_parking import CashParking, ParkingConfig
    from strategy.fractional.momentum_rotation import MomentumRotation, MomentumConfig

    frac_cfg = load_fractional_config()
    if not frac_cfg.get("enabled", False):
        print("  ETF 策略未启用")
        return

    notifier = None
    try:
        from notification.telegram_bot import TelegramNotifier
        notifier = TelegramNotifier()
    except Exception:
        pass

    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    trade_ctx = OpenSecTradeContext(host="127.0.0.1", port=11111)

    try:
        # ── Cash Parking ──
        if frac_cfg.get("cash_parking", {}).get("enabled", False):
            print("\n  [Cash Parking] 检查闲置资金...")
            park_cfg = ParkingConfig.from_yaml(frac_cfg)
            park_cfg.trade_env = trade_env
            parking = CashParking(park_cfg, quote_ctx, trade_ctx, notifier)

            if dry_run:
                status = parking.status()
                print(f"    现金: ${status['cash']:,.2f} | "
                      f"SGOV: {status['shares']}股 (${status['value']:,.0f}) | "
                      f"闲置: ${status['idle']:,.0f}")
            else:
                result = parking.check_and_park()
                if result:
                    print(f"    操作: {result['action']} {result['symbol']} x {result['qty']}")
                else:
                    status = parking.status()
                    print(f"    无需操作 (现金=${status['cash']:,.0f}, "
                          f"SGOV={status['shares']}股)")

        # ── Momentum Rotation ──
        if frac_cfg.get("momentum", {}).get("enabled", False):
            mom_cfg = MomentumConfig.from_yaml(frac_cfg)
            mom_cfg.trade_env = trade_env
            momentum = MomentumRotation(mom_cfg, quote_ctx, trade_ctx, notifier)

            if momentum.is_rebalance_day():
                print("\n  [动量轮动] 今日是再平衡日!")
                if dry_run:
                    print("    (干跑模式，不执行)")
                    momentum.rebalance(dry_run=True)
                else:
                    momentum.rebalance(dry_run=False)
            else:
                status = momentum.status()
                print(f"\n  [动量轮动] 非再平衡日 | "
                      f"持仓: ${status['total_value']:,.0f}/{status['budget']}")

    finally:
        quote_ctx.close()
        trade_ctx.close()


def send_daily_report():
    """Build and send end-of-day summary via Telegram. Never raises."""
    try:
        from datetime import datetime, timezone, timedelta
        from data.trade_store import TradeStore
        from notification.telegram_bot import TelegramNotifier

        try:
            import zoneinfo
            et = zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            et = timezone(timedelta(hours=-4))
        today_str = datetime.now(et).strftime("%Y-%m-%d")

        store = TradeStore()

        stock_trades = store.query_trades(days=1)
        option_trades = store.query_option_trades(days=1)
        open_options = store.query_option_trades(status="open")
        pdt_used = store.count_recent_day_trades(days=5)

        stock_count = len(stock_trades)
        option_count = len(option_trades)

        realized_pnl = 0.0
        for t in stock_trades:
            realized_pnl += t.get("pnl", 0.0) or 0.0
        for t in option_trades:
            if t.get("status") == "closed":
                realized_pnl += t.get("realized_pnl", 0.0) or 0.0

        position_lines = []
        for pos in open_options:
            legs_str = pos.get("legs", "[]")
            try:
                import json
                leg_count = len(json.loads(legs_str))
            except Exception:
                leg_count = legs_str.count("code")
            position_lines.append(
                f"  {pos.get('strategy', '?')} "
                f"{pos.get('underlying', '?')} "
                f"legs={leg_count}"
            )
        if not position_lines:
            position_block = "  无"
        else:
            position_block = "\n".join(position_lines)

        frac_cfg = load_fractional_config()
        mom_enabled = frac_cfg.get("momentum", {}).get("enabled", False)
        if mom_enabled:
            now_et = datetime.now(et)
            is_rebal = now_et.day <= 5 and now_et.weekday() < 5
            mom_status = "再平衡日" if is_rebal else "非再平衡日"
        else:
            mom_status = "未启用"

        cs_max = 3
        try:
            from run_options import load_options_config
            opts_cfg = load_options_config()
            cs_max = opts_cfg.get("credit_spread", {}).get("max_positions", 3)
        except Exception:
            pass

        cs_open = sum(1 for p in open_options
                      if p.get("strategy") == "credit_spread")
        cs_status = "已满仓" if cs_open >= cs_max else f"扫描中 ({cs_open}/{cs_max})"

        pnl_sign = "+" if realized_pnl >= 0 else ""
        report = (
            f"📊 <b>FUTU-QUANT 每日报告</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📅 日期: {today_str}\n"
            f"\n"
            f"📈 <b>今日交易</b>\n"
            f"  期权: {option_count} 笔 | 股票: {stock_count} 笔\n"
            f"  已实现 P&amp;L: {pnl_sign}${abs(realized_pnl):,.2f}\n"
            f"\n"
            f"📦 <b>当前持仓</b>\n"
            f"  期权: {len(open_options)} 个 open\n"
            f"{position_block}\n"
            f"\n"
            f"⚠️ <b>PDT 状态</b>\n"
            f"  5日内已用: {pdt_used}/3\n"
            f"\n"
            f"📋 <b>明日计划</b>\n"
            f"  - 动量轮动: {mom_status}\n"
            f"  - Credit Spread: {cs_status}"
        )

        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        notifier = TelegramNotifier(tg_token, tg_chat, enabled=bool(tg_token and tg_chat))
        notifier.notify_daily_report(report)

        store.close()
        print("[Daily Report] 每日报告已推送")
    except Exception as e:
        print(f"[Daily Report] 推送失败: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FUTU-QUANT 一键启动")
    parser.add_argument("--dry-run", action="store_true", help="干跑模式")
    parser.add_argument("--status", action="store_true", help="查看持仓状态")
    parser.add_argument("--force", action="store_true", help="忽略交易时间检查")
    args = parser.parse_args()

    from run_options import load_options_config, run_live, show_status

    config = load_options_config()
    options_enabled = config.get("enabled", False)

    if args.status:
        if options_enabled:
            show_status(config)
        else:
            print("期权策略已暂停。仅运行 ETF 动量轮动。")
        return

    # Prevent multiple instances
    lock_path = _acquire_lock()

    # Global trade lock — block if another trading script is running
    from utils.trade_lock import acquire_trade_lock
    acquire_trade_lock("start")

    market_open = is_us_market_open()

    if not market_open and not args.dry_run:
        from datetime import datetime, timezone, timedelta
        try:
            import zoneinfo
            et = zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            et = timezone(timedelta(hours=-4))
        now_et = datetime.now(et)

        if not args.force:
            print(f"美股未开盘 (当前美东时间 {now_et.strftime('%H:%M %A')})")
            print(f"交易时间: 周一至周五 9:30-16:00 ET (北京时间 21:30-04:00)")
            print(f"\n如需强制运行: python start.py --force")
            print(f"干跑模式(不下单): python start.py --dry-run")
            return

        # --force but market closed: wait for market open
        print(f"美股未开盘 (美东 {now_et.strftime('%H:%M')})", flush=True)
        print("--force 模式: 等待开盘后自动启动...", flush=True)
        import time as _time
        while not is_us_market_open():
            _time.sleep(30)
            now_et = datetime.now(et)
            print(f"  等待中... 美东 {now_et.strftime('%H:%M:%S')}", flush=True)
        print(f"美股已开盘! 启动交易系统...", flush=True)

    dry_run = args.dry_run
    trade_env = config.get("trade_env", "REAL")

    try:
        # ── ETF strategies first (Cash Parking + Momentum) ──
        print("\n" + "=" * 60)
        print("  ETF 策略 (Cash Parking + 动量轮动)")
        print("=" * 60)
        try:
            run_etf_strategies(dry_run=dry_run, trade_env=trade_env)
        except Exception as e:
            print(f"  ETF 策略异常: {e}")

        # ── Start Telegram interactive bot (background) ──
        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        tg_bot = None
        if tg_token and tg_chat:
            try:
                from notification.telegram_bot import TelegramCommandBot
                tg_bot = TelegramCommandBot(tg_token, tg_chat)
                tg_bot.start_polling()
                print("  [Telegram Bot] 交互式命令已启动")
            except Exception as e:
                print(f"  [Telegram Bot] 启动失败: {e}")

        # ── Factor library market timing ──
        try:
            from factor_library.storage import load_factors
            from factor_library.search import build_factor_matrix
            from factor_library.screener import market_timing_signal

            factor_dfs = {}
            for cat in ["technical", "risk", "volatility", "liquidity"]:
                df = load_factors(cat)
                if not df.empty:
                    factor_dfs[cat] = df
            if factor_dfs:
                matrix = build_factor_matrix(factor_dfs)
                if not matrix.empty:
                    timing = market_timing_signal(matrix)
                    state = timing["market_state"]
                    score = timing["score"]
                    print(f"\n  [因子择时] 市场状态: {state} (score={score})")
                    print(f"  建议: {timing['recommendation']}")
                    if state == "BEARISH" and not dry_run:
                        print("  ⚠ 熊市信号 — 期权策略将降低开仓数量")
        except Exception as e:
            print(f"  [因子择时] 不可用 ({e})")

        # ── Options strategies (only if enabled) ──
        if options_enabled:
            print("\n" + "=" * 60)
            print("  期权策略 (Credit Spread)")
            print("=" * 60)
            strategies = ["credit_spread"]
            run_live(strategies, dry_run, config)
        else:
            print("\n  [期权] 已暂停 (本金不足，等 $5,000+ 再启用)")

        # End-of-day report after trading session completes
        send_daily_report()
    finally:
        if tg_bot:
            tg_bot.stop_polling()
        _release_lock(lock_path)


if __name__ == "__main__":
    main()
