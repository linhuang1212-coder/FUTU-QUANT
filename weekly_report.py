"""
FUTU-QUANT 周末自动复盘 — 每周六自动生成并推送周报

Usage:
    python weekly_report.py              # 生成周报 + Telegram 推送 + 保存 Markdown
    python weekly_report.py --dry-run    # 仅打印，不推送 Telegram
    python weekly_report.py --install    # 注册 Windows 定时任务 (每周六 10:00)
"""
from __future__ import annotations

import sys
import os
import io
import json
import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load .env
_root = Path(__file__).resolve().parent
_env_path = _root / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _et_now() -> datetime:
    """Current time in US/Eastern."""
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        et = timezone(timedelta(hours=-4))
    return datetime.now(et)


def _iso_week_label(dt: datetime) -> str:
    """Return 'YYYY-WNN' ISO week label."""
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _load_fractional_config() -> dict:
    from utils.helpers import load_yaml
    cfg_path = _root / "config" / "fractional.yaml"
    if cfg_path.exists():
        return load_yaml(str(cfg_path)).get("fractional", {})
    return {}


def _load_options_config() -> dict:
    from utils.helpers import load_yaml
    cfg_path = _root / "config" / "options.yaml"
    if cfg_path.exists():
        return load_yaml(str(cfg_path)).get("options", {})
    return {}


# ── Data Collection ──────────────────────────────────────────


def collect_weekly_data() -> dict:
    """Query last 7 days of trades and option_trades, compute statistics."""
    from data.trade_store import TradeStore

    store = TradeStore()
    try:
        stock_trades = store.query_trades(days=7, limit=1000)
        option_trades = store.query_option_trades(days=7, limit=1000)
        open_options = store.query_option_trades(status="open", limit=200)
    finally:
        store.close()

    closed_options = [t for t in option_trades if t.get("status") == "closed"]

    stock_pnl = sum(t.get("pnl", 0.0) or 0.0 for t in stock_trades)
    option_pnl = sum(t.get("realized_pnl", 0.0) or 0.0 for t in closed_options)
    total_pnl = stock_pnl + option_pnl

    total_count = len(stock_trades) + len(option_trades)

    wins = sum(1 for t in stock_trades if (t.get("pnl", 0) or 0) > 0)
    wins += sum(1 for t in closed_options if (t.get("realized_pnl", 0) or 0) > 0)
    closed_count = len(stock_trades) + len(closed_options)
    win_rate = (wins / closed_count * 100) if closed_count > 0 else 0.0

    all_pnls = [t.get("pnl", 0) or 0 for t in stock_trades]
    all_pnls += [t.get("realized_pnl", 0) or 0 for t in closed_options]
    max_win = max(all_pnls) if all_pnls else 0.0
    max_loss = min(all_pnls) if all_pnls else 0.0

    # Per-strategy breakdown
    by_strategy: dict[str, dict] = {}
    for t in option_trades:
        strat = t.get("strategy", "unknown")
        entry = by_strategy.setdefault(strat, {"count": 0, "pnl": 0.0, "open": 0, "closed": 0})
        entry["count"] += 1
        if t.get("status") == "closed":
            entry["pnl"] += t.get("realized_pnl", 0.0) or 0.0
            entry["closed"] += 1
        else:
            entry["open"] += 1

    for t in stock_trades:
        strat = t.get("strategy", "") or "stock"
        entry = by_strategy.setdefault(strat, {"count": 0, "pnl": 0.0, "open": 0, "closed": 0})
        entry["count"] += 1
        entry["pnl"] += t.get("pnl", 0.0) or 0.0
        entry["closed"] += 1

    # Opened / closed this week
    opened_this_week = [t for t in option_trades if t.get("status") == "open"]
    closed_this_week = closed_options

    # Momentum rotation holdings
    momentum_holdings = _get_momentum_holdings(stock_trades)

    # SGOV (cash parking) holdings
    sgov_qty = _get_sgov_qty(stock_trades)

    return {
        "total_count": total_count,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_win": max_win,
        "max_loss": max_loss,
        "by_strategy": by_strategy,
        "open_options": open_options,
        "opened_this_week": opened_this_week,
        "closed_this_week": closed_this_week,
        "momentum_holdings": momentum_holdings,
        "sgov_qty": sgov_qty,
        "stock_trades": stock_trades,
        "option_trades": option_trades,
    }


def _get_momentum_holdings(stock_trades: list[dict]) -> list[str]:
    """Infer momentum rotation holdings from recent stock trades."""
    frac_cfg = _load_fractional_config()
    pool = frac_cfg.get("momentum", {}).get("pool", [])
    pool_tickers = [s.replace("US.", "") for s in pool]

    held: set[str] = set()
    for t in stock_trades:
        sym = t.get("symbol", "").replace("US.", "")
        if sym in pool_tickers:
            action = t.get("action", "").upper()
            if action == "BUY":
                held.add(sym)
            elif action == "SELL" and sym in held:
                held.discard(sym)
    return sorted(held)


def _get_sgov_qty(stock_trades: list[dict]) -> int:
    """Infer net SGOV shares from recent trades."""
    qty = 0
    for t in stock_trades:
        sym = t.get("symbol", "").replace("US.", "")
        if sym in ("SGOV", "BIL"):
            action = t.get("action", "").upper()
            q = t.get("qty", 0) or 0
            if action == "BUY":
                qty += q
            elif action == "SELL":
                qty -= q
    return max(qty, 0)


# ── Report Formatting ────────────────────────────────────────


def build_telegram_report(data: dict, week_label: str) -> str:
    """Build HTML-formatted Telegram message."""
    pnl = data["total_pnl"]
    pnl_sign = "+" if pnl >= 0 else ""

    # Strategy breakdown
    strategy_lines = []

    cs = data["by_strategy"].get("credit_spread", {})
    if cs.get("count", 0) > 0:
        cs_pnl = cs.get("pnl", 0)
        strategy_lines.append(
            f"  Credit Spread: {cs['count']} 笔, P&amp;L {'+' if cs_pnl >= 0 else ''}${cs_pnl:,.2f}"
        )

    mom_holdings = data["momentum_holdings"]
    if mom_holdings:
        strategy_lines.append(f"  动量轮动: 持仓 [{', '.join(mom_holdings)}]")
    else:
        strategy_lines.append("  动量轮动: 无持仓")

    sgov = data["sgov_qty"]
    strategy_lines.append(f"  Cash Parking: SGOV {sgov} 股")

    for strat, info in data["by_strategy"].items():
        if strat in ("credit_spread", "momentum", "cash_parking", "stock"):
            continue
        if info["count"] > 0:
            s_pnl = info.get("pnl", 0)
            strategy_lines.append(
                f"  {strat}: {info['count']} 笔, P&amp;L {'+' if s_pnl >= 0 else ''}${s_pnl:,.2f}"
            )

    strategy_block = "\n".join(strategy_lines) if strategy_lines else "  无"

    # Open positions
    position_lines = []
    for pos in data["open_options"]:
        underlying = pos.get("underlying", "?")
        strat = pos.get("strategy", "?")
        max_loss = pos.get("max_loss", 0)
        ts = pos.get("timestamp", "")[:10]
        position_lines.append(f"  [{strat}] {underlying} max_loss=${max_loss:.0f} ({ts})")

    if mom_holdings:
        position_lines.append(f"  [动量轮动] {', '.join(mom_holdings)}")
    if sgov > 0:
        position_lines.append(f"  [Cash Parking] SGOV x{sgov}")

    position_block = "\n".join(position_lines) if position_lines else "  无"

    # Next week outlook
    frac_cfg = _load_fractional_config()
    mom_cfg = frac_cfg.get("momentum", {})
    now_et = _et_now()
    next_monday = now_et + timedelta(days=(7 - now_et.weekday()) % 7 or 7)
    is_rebalance = mom_cfg.get("enabled", False) and next_monday.day <= 5

    opts_cfg = _load_options_config()
    cs_max = opts_cfg.get("credit_spread", {}).get("max_positions", 3)
    cs_open = sum(1 for p in data["open_options"]
                  if p.get("strategy") == "credit_spread")
    cs_slots = max(0, cs_max - cs_open)

    report = (
        f"📊 <b>FUTU-QUANT 周报</b> ({week_label})\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📈 <b>本周交易汇总</b>\n"
        f"  总交易: {data['total_count']} 笔 | 胜率: {data['win_rate']:.0f}%\n"
        f"  已实现 P&amp;L: {pnl_sign}${abs(pnl):,.2f}\n"
        f"  最大单笔盈利: ${data['max_win']:+,.2f}\n"
        f"  最大单笔亏损: ${data['max_loss']:+,.2f}\n"
        f"\n"
        f"💰 <b>策略表现</b>\n"
        f"{strategy_block}\n"
        f"\n"
        f"📦 <b>当前持仓</b>\n"
        f"{position_block}\n"
        f"\n"
        f"📋 <b>下周展望</b>\n"
        f"  - 再平衡日: {'是' if is_rebalance else '否'}\n"
        f"  - Credit Spread 名额: {cs_slots}/{cs_max}"
    )
    return report


def build_markdown_report(data: dict, week_label: str) -> str:
    """Build Markdown report for docs/weekly/YYYY-WNN.md."""
    now_str = _et_now().strftime("%Y-%m-%d %H:%M ET")
    pnl = data["total_pnl"]

    lines = [
        f"# FUTU-QUANT 周报 {week_label}",
        f"",
        f"生成时间: {now_str}",
        f"",
        f"## 本周交易汇总",
        f"",
        f"| 指标 | 值 |",
        f"|------|------|",
        f"| 总交易笔数 | {data['total_count']} |",
        f"| 胜率 | {data['win_rate']:.1f}% |",
        f"| 已实现 P&L | ${pnl:+,.2f} |",
        f"| 最大单笔盈利 | ${data['max_win']:+,.2f} |",
        f"| 最大单笔亏损 | ${data['max_loss']:+,.2f} |",
        f"",
        f"## 策略分类表现",
        f"",
        f"| 策略 | 笔数 | 已平仓 | P&L |",
        f"|------|------|--------|------|",
    ]

    for strat, info in data["by_strategy"].items():
        lines.append(
            f"| {strat} | {info['count']} | {info['closed']} | ${info['pnl']:+,.2f} |"
        )

    lines += [
        f"",
        f"## 本周开仓",
        f"",
    ]
    if data["opened_this_week"]:
        lines.append("| 标的 | 策略 | Max Loss | 时间 |")
        lines.append("|------|------|----------|------|")
        for t in data["opened_this_week"]:
            lines.append(
                f"| {t.get('underlying', '?')} | {t.get('strategy', '?')} "
                f"| ${t.get('max_loss', 0):,.0f} | {t.get('timestamp', '')[:16]} |"
            )
    else:
        lines.append("无")

    lines += [
        f"",
        f"## 本周平仓",
        f"",
    ]
    if data["closed_this_week"]:
        lines.append("| 标的 | 策略 | P&L | 原因 | 时间 |")
        lines.append("|------|------|------|------|------|")
        for t in data["closed_this_week"]:
            lines.append(
                f"| {t.get('underlying', '?')} | {t.get('strategy', '?')} "
                f"| ${t.get('realized_pnl', 0):+,.2f} | {t.get('close_reason', '')} "
                f"| {t.get('close_timestamp', '')[:16]} |"
            )
    else:
        lines.append("无")

    lines += [
        f"",
        f"## 当前持仓",
        f"",
    ]
    if data["open_options"]:
        lines.append("| 标的 | 策略 | Max Loss | 开仓时间 |")
        lines.append("|------|------|----------|----------|")
        for p in data["open_options"]:
            lines.append(
                f"| {p.get('underlying', '?')} | {p.get('strategy', '?')} "
                f"| ${p.get('max_loss', 0):,.0f} | {p.get('timestamp', '')[:16]} |"
            )
    else:
        lines.append("无")

    if data["momentum_holdings"]:
        lines.append(f"\n动量轮动持仓: {', '.join(data['momentum_holdings'])}")
    if data["sgov_qty"] > 0:
        lines.append(f"\nCash Parking: SGOV x{data['sgov_qty']}")

    lines.append("")
    return "\n".join(lines)


# ── Save & Send ──────────────────────────────────────────────


def save_markdown(md_content: str, week_label: str) -> Path:
    """Save report to docs/weekly/YYYY-WNN.md."""
    docs_dir = _root / "docs" / "weekly"
    docs_dir.mkdir(parents=True, exist_ok=True)
    filepath = docs_dir / f"{week_label}.md"
    filepath.write_text(md_content, encoding="utf-8")
    return filepath


def send_telegram(html_report: str) -> bool:
    """Push report via Telegram."""
    from notification.telegram_bot import TelegramNotifier
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not tg_token or not tg_chat:
        print("[Weekly] Telegram 未配置 (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return False
    notifier = TelegramNotifier(tg_token, tg_chat, enabled=True)
    return notifier.send_sync(html_report)


# ── Install Windows Scheduled Task ──────────────────────────


def install_scheduled_task():
    """Register weekly report as a Windows Task Scheduler task (Saturday 10:00)."""
    bat_path = _root / "scheduled_weekly.bat"
    if not bat_path.exists():
        print(f"[Install] 找不到 {bat_path}")
        print("[Install] 请确保 scheduled_weekly.bat 存在")
        return False

    task_name = "FUTU-QUANT-WeeklyReport"
    cmd = [
        "schtasks", "/Create",
        "/TN", task_name,
        "/TR", f'"{bat_path}"',
        "/SC", "WEEKLY",
        "/D", "SAT",
        "/ST", "10:00",
        "/F",
    ]

    print(f"[Install] 注册定时任务: {task_name}")
    print(f"[Install] 命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if result.returncode == 0:
            print(f"[Install] 成功! 每周六 10:00 自动运行周报")
            print(f"[Install] 管理: schtasks /Query /TN {task_name}")
            print(f"[Install] 删除: schtasks /Delete /TN {task_name} /F")
            return True
        else:
            print(f"[Install] 失败: {result.stderr.strip()}")
            print("[Install] 请以管理员权限运行")
            return False
    except Exception as e:
        print(f"[Install] 异常: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────


def generate_weekly_report(dry_run: bool = False) -> None:
    """Full pipeline: collect data -> build reports -> save + send."""
    now_et = _et_now()
    week_label = _iso_week_label(now_et)

    print(f"[Weekly] 生成周报 {week_label} ...")

    data = collect_weekly_data()

    tg_report = build_telegram_report(data, week_label)
    md_report = build_markdown_report(data, week_label)

    md_path = save_markdown(md_report, week_label)
    print(f"[Weekly] Markdown 已保存: {md_path}")

    if dry_run:
        print(f"\n{'=' * 60}")
        print(tg_report)
        print(f"{'=' * 60}")
        print("[Weekly] 干跑模式，未推送 Telegram")
    else:
        ok = send_telegram(tg_report)
        if ok:
            print("[Weekly] Telegram 推送成功")
        else:
            print("[Weekly] Telegram 推送失败 (报告已保存到 Markdown)")

    print(f"[Weekly] 完成 — 交易 {data['total_count']} 笔, "
          f"P&L ${data['total_pnl']:+,.2f}, 胜率 {data['win_rate']:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="FUTU-QUANT 周报生成器")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印报告，不推送 Telegram")
    parser.add_argument("--install", action="store_true",
                        help="注册 Windows 定时任务 (每周六 10:00)")
    args = parser.parse_args()

    if args.install:
        install_scheduled_task()
        return

    try:
        generate_weekly_report(dry_run=args.dry_run)
    except Exception as e:
        print(f"[Weekly] 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
