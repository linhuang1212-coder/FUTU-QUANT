"""
ETF 持仓实时监控 — 持久化止损止盈 + 自动轮换

核心设计:
  - SQLite 持久化: 所有持仓记录在 data_store/etf_monitor.db
  - 程序重启不丢失: 成本价/数量/买入时间全部从数据库恢复
  - 止损止盈基于数据库成本价, 不依赖富途 API 的 cost_price
  - 触发后自动扫描替补, 有合格标的就买入, 没有就闲置

Usage:
  python run_monitor.py              # 实盘监控
  python run_monitor.py --dry-run    # 只看不操作
  python run_monitor.py --tp 0.15    # 止盈 15%
  python run_monitor.py --sl 0.08    # 止损 8%
  python run_monitor.py --init       # 从富途同步当前持仓到数据库
"""
from __future__ import annotations

import sys
import io
import os
import time
import sqlite3
import argparse
import atexit
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                               errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                               errors="replace", line_buffering=True)

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from utils.logger import setup_logger
from utils.helpers import load_yaml

logger = setup_logger("monitor_etf")

_frac_cfg = load_yaml(str(Path(__file__).resolve().parent / "config" / "fractional.yaml"))
_mom_cfg = _frac_cfg.get("fractional", {}).get("momentum", {})
ETF_POOL = _mom_cfg.get("pool", [])

import re

DB_PATH = Path(__file__).resolve().parent / "data_store" / "etf_monitor.db"
LOCK_PATH = Path(__file__).resolve().parent / ".monitor.lock"

PNL_SANITY_THRESHOLD = 0.50  # >50% PnL in a single check = likely data error

# Match option symbols: US.SLV260529C74000, US.GDX260605P95000, etc.
_OPTION_RE = re.compile(r"\d{6}[CP]\d")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Instance lock — prevent multiple monitors from running
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _kill_other_monitors():
    """Kill any other run_monitor.py processes before starting."""
    try:
        import psutil
        my_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] == my_pid:
                    continue
                cmdline = proc.info.get("cmdline") or []
                cmd_str = " ".join(cmdline).lower()
                if "run_monitor" in cmd_str and "python" in (proc.info.get("name") or "").lower():
                    print(f"  [LOCK] Killing old monitor process PID={proc.info['pid']}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        pass


def _acquire_monitor_lock():
    _kill_other_monitors()
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text().strip())
            import psutil
            if psutil.pid_exists(old_pid) and old_pid != os.getpid():
                proc = psutil.Process(old_pid)
                if proc.is_running() and "python" in proc.name().lower():
                    print(f"  [LOCK] Force-killing stale monitor PID={old_pid}")
                    proc.kill()
        except (ImportError, ValueError, Exception):
            pass
    LOCK_PATH.write_text(str(os.getpid()))
    atexit.register(_release_monitor_lock)


def _release_monitor_lock():
    try:
        if LOCK_PATH.exists():
            pid_in_file = int(LOCK_PATH.read_text().strip())
            if pid_in_file == os.getpid():
                LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persistent Position Store
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PositionStore:
    """SQLite-backed position tracking — survives restarts."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    qty INTEGER NOT NULL,
                    cost_price REAL NOT NULL,
                    bought_at TEXT NOT NULL,
                    strategy TEXT DEFAULT 'momentum_rotation'
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    price REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    reason TEXT DEFAULT ''
                )
            """)

    def get_positions(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM positions WHERE qty > 0").fetchall()
            return [dict(r) for r in rows]

    def get_position(self, symbol: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM positions WHERE symbol=?",
                            (symbol,)).fetchone()
            return dict(row) if row else None

    def add_position(self, symbol: str, qty: int, cost_price: float,
                     strategy: str | None = None):
        now = datetime.now().isoformat()
        if strategy is None:
            strategy = "option_call" if _OPTION_RE.search(symbol) else "momentum_rotation"
        with self._conn() as c:
            existing = c.execute("SELECT qty, cost_price FROM positions WHERE symbol=?",
                                 (symbol,)).fetchone()
            if existing and existing["qty"] > 0:
                old_qty = existing["qty"]
                old_cost = existing["cost_price"]
                new_qty = old_qty + qty
                avg_cost = (old_qty * old_cost + qty * cost_price) / new_qty
                c.execute("UPDATE positions SET qty=?, cost_price=?, bought_at=? WHERE symbol=?",
                          (new_qty, round(avg_cost, 4), now, symbol))
            else:
                c.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?)",
                          (symbol, qty, round(cost_price, 4), now, strategy))
            c.execute("INSERT INTO trade_log (timestamp, action, symbol, qty, price) VALUES (?,?,?,?,?)",
                      (now, "BUY", symbol, qty, cost_price))
        logger.info(f"[DB] Added {symbol} x{qty} @ ${cost_price:.2f}")

    def remove_position(self, symbol: str, sell_price: float, reason: str = ""):
        with self._conn() as c:
            pos = c.execute("SELECT qty, cost_price FROM positions WHERE symbol=?",
                            (symbol,)).fetchone()
            if not pos:
                return
            qty = pos["qty"]
            cost = pos["cost_price"]
            pnl = (sell_price - cost) * qty
            now = datetime.now().isoformat()
            c.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
            c.execute("INSERT INTO trade_log (timestamp, action, symbol, qty, price, pnl, reason) "
                      "VALUES (?,?,?,?,?,?,?)",
                      (now, "SELL", symbol, qty, sell_price, round(pnl, 2), reason))
        logger.info(f"[DB] Removed {symbol} x{qty} @ ${sell_price:.2f} "
                    f"PnL=${pnl:+.2f} ({reason})")

    def get_trade_log(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM trade_log ORDER BY id DESC LIMIT ?",
                             (limit,)).fetchall()
            return [dict(r) for r in rows]

    def sync_from_broker(self, trade_ctx):
        """One-time sync: import current broker positions into DB."""
        from futu import TrdEnv, RET_OK
        ret, positions = trade_ctx.position_list_query(trd_env=TrdEnv.REAL)
        if ret != RET_OK or positions is None or positions.empty:
            logger.warning("[SYNC] No broker positions found")
            return

        count = 0
        for _, pos in positions.iterrows():
            qty = int(pos.get("qty", 0))
            if qty <= 0:
                continue
            code = str(pos.get("code", ""))
            cost = float(pos.get("cost_price", 0))
            if cost <= 0:
                cost = float(pos.get("nominal_price", 0))
            if cost <= 0:
                continue

            existing = self.get_position(code)
            if existing:
                logger.info(f"[SYNC] {code} already in DB, skip")
                continue

            self.add_position(code, qty, cost)
            count += 1

        logger.info(f"[SYNC] Imported {count} positions from broker")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Market scanning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def is_market_open() -> bool:
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        et = timezone(timedelta(hours=-4))
    now_et = datetime.now(et)
    if now_et.weekday() >= 5:
        return False
    t = now_et.hour * 60 + now_et.minute
    return 9 * 60 + 30 <= t <= 16 * 60


def scan_momentum(quote_ctx, held_symbols: set[str]) -> list[dict]:
    """Scan ETF pool for replacement candidates."""
    from futu import RET_OK, KLType, SubType
    import numpy as np

    candidates = []
    for sym in ETF_POOL:
        if sym in held_symbols:
            continue
        try:
            quote_ctx.subscribe([sym], [SubType.K_DAY])
            ret, klines = quote_ctx.get_cur_kline(sym, 300, KLType.K_DAY)
            if ret != RET_OK or klines.empty or len(klines) < 252:
                continue

            closes = klines["close"].values
            price = float(closes[-1])
            price_12m = float(closes[-253]) if len(closes) > 253 else float(closes[0])
            price_1m = float(closes[-22]) if len(closes) > 22 else price
            momentum = (price_1m / price_12m) - 1.0
            sma200 = float(np.mean(closes[-200:]))

            if price <= sma200:
                continue

            candidates.append({
                "symbol": sym, "momentum": momentum,
                "price": price, "above_sma": True,
            })
            time.sleep(0.3)
        except Exception:
            continue

    candidates.sort(key=lambda x: x["momentum"], reverse=True)
    return candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trade execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def execute_sell(trade_ctx, code: str, qty: int, price: float,
                 dry_run: bool, reason: str = "") -> bool:
    from futu import TrdSide, OrderType, TrdEnv, RET_OK

    # Final safety gate: log every sell attempt with full context
    is_opt = bool(_OPTION_RE.search(code))
    logger.info(f"[SELL-GATE] code={code} qty={qty} price=${price:.2f} "
                f"is_option={is_opt} reason={reason} dry_run={dry_run}")

    if dry_run:
        logger.info(f"  [DRY] SELL {code} x{qty} @ ${price:.2f}")
        return True

    sell_price = round(price * 0.995, 2)
    ret, data = trade_ctx.place_order(
        price=sell_price, qty=qty, code=code,
        trd_side=TrdSide.SELL, order_type=OrderType.NORMAL,
        trd_env=TrdEnv.REAL,
    )
    if ret == RET_OK:
        logger.info(f"  SELL {code} x{qty} @ ${sell_price:.2f} -> OK")
        return True
    logger.error(f"  SELL {code} FAILED: {data}")
    return False


def execute_buy(trade_ctx, quote_ctx, symbol: str, budget_usd: float,
                dry_run: bool) -> tuple[bool, int, float]:
    """Returns (success, qty, price)."""
    from futu import TrdSide, OrderType, TrdEnv, RET_OK

    ret, snap = quote_ctx.get_market_snapshot([symbol])
    if ret != RET_OK or snap is None or snap.empty:
        return False, 0, 0

    price = round(float(snap.iloc[0].get("last_price", 0)), 2)
    if price <= 0:
        return False, 0, 0

    qty = int(budget_usd / price)
    if qty < 1:
        logger.info(f"  {symbol} @ ${price:.2f} too expensive for ${budget_usd:.0f}")
        return False, 0, 0

    if dry_run:
        logger.info(f"  [DRY] BUY {symbol} x{qty} @ ${price:.2f}")
        return True, qty, price

    ret, data = trade_ctx.place_order(
        price=price, qty=qty, code=symbol,
        trd_side=TrdSide.BUY, order_type=OrderType.NORMAL,
        trd_env=TrdEnv.REAL,
    )
    if ret == RET_OK:
        logger.info(f"  BUY {symbol} x{qty} @ ${price:.2f} -> OK")
        return True, qty, price
    logger.error(f"  BUY {symbol} FAILED: {data}")
    return False, 0, 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main monitor loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_monitor(take_profit: float = 0.15,
                stop_loss: float = 0.08,
                interval: int = 60,
                dry_run: bool = False):
    from futu import OpenQuoteContext, OpenSecTradeContext, TrdEnv, TrdMarket, RET_OK

    _acquire_monitor_lock()

    # Global trade lock — block if another trading script is running
    from utils.trade_lock import acquire_trade_lock
    acquire_trade_lock("run_monitor")

    store = PositionStore()
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    trade_ctx = OpenSecTradeContext(host="127.0.0.1", port=11111,
                                    filter_trdmarket=TrdMarket.US)

    notifier = None
    try:
        from notification.telegram_bot import TelegramNotifier
        notifier = TelegramNotifier()
    except Exception:
        pass

    def notify(msg: str):
        if notifier:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(notifier.send_message(msg))
                loop.close()
            except Exception:
                pass

    # Load persisted positions
    db_positions = store.get_positions()
    mode = "DRY-RUN" if dry_run else "LIVE"

    n_opts = sum(1 for p in db_positions if _OPTION_RE.search(p["symbol"]))
    n_etfs = len(db_positions) - n_opts
    print(f"\n{'='*60}")
    print(f"  ETF+Options Monitor ({mode}) — Persistent Mode")
    print(f"  ETF  TP: +{take_profit:.0%} | SL: -{stop_loss:.0%}")
    print(f"  OPT  TP: +50% | SL: -40% (DTE<=5: -30%)")
    print(f"  Interval: {interval}s | DB: {DB_PATH.name}")
    print(f"  Tracked: {n_etfs} ETFs + {n_opts} Options")
    for p in db_positions:
        is_opt = bool(_OPTION_RE.search(p["symbol"]))
        label = "OPT" if is_opt else "ETF"
        print(f"    [{label}] {p['symbol']:28s} x{p['qty']:2d} @ ${p['cost_price']:.2f} "
              f"(since {p['bought_at'][:10]})")
    print(f"{'='*60}\n")

    notify(f"<b>ETF Monitor Started</b>\n"
           f"Mode: {mode} | TP: +{take_profit:.0%} | SL: -{stop_loss:.0%}\n"
           f"Positions: {len(db_positions)}")

    scan_count = 0

    try:
        while True:
            if not is_market_open():
                if scan_count == 0:
                    print(f"  [{_now()}] Market closed, waiting...")
                time.sleep(300)
                continue

            scan_count += 1
            db_positions = store.get_positions()
            if not db_positions:
                if scan_count % 10 == 1:
                    print(f"  [{_now()}] No tracked positions")
                time.sleep(interval)
                continue

            # Get live prices for all tracked symbols
            symbols = [p["symbol"] for p in db_positions]
            try:
                ret, snap = quote_ctx.get_market_snapshot(symbols)
                if ret != RET_OK or snap is None or snap.empty:
                    time.sleep(interval)
                    continue
                live_prices = {}
                for _, row in snap.iterrows():
                    live_prices[row["code"]] = float(row.get("last_price", 0))
            except Exception as e:
                logger.warning(f"  Price fetch error: {e}")
                time.sleep(interval)
                continue

            triggered = []

            for p in db_positions:
                sym = p["symbol"]
                qty = p["qty"]
                cost = p["cost_price"]
                strategy = p.get("strategy", "momentum_rotation")
                # Detect options: check strategy field, or symbol pattern
                # Option symbols look like US.SLV260529C74000 or US.GDX260605P95000
                # They contain C or P followed by strike price after a date
                is_option = (
                    strategy == "option_call"
                    or strategy == "option_put"
                    or bool(_OPTION_RE.search(sym))
                )
                cur_price = live_prices.get(sym, 0)

                if cur_price <= 0:
                    continue

                if cost <= 0:
                    logger.warning(f"  {sym} cost_price={cost} invalid, SKIP")
                    continue

                pnl_ratio = (cur_price / cost) - 1.0
                pnl_val = (cur_price - cost) * qty
                if is_option:
                    pnl_val *= 100  # 1 contract = 100 shares

                # Options: higher sanity threshold (can move 200%+ in a day)
                sanity = 5.0 if is_option else PNL_SANITY_THRESHOLD
                if abs(pnl_ratio) > sanity:
                    logger.warning(
                        f"  {sym} PnL={pnl_ratio:+.1%} exceeds sanity "
                        f"threshold ({sanity:.0%}), "
                        f"cost=${cost:.2f} now=${cur_price:.2f} — SKIP")
                    continue

                # Different TP/SL for options vs stocks
                if is_option:
                    opt_tp = 0.50    # +50% take profit for options
                    opt_sl = 0.40    # -40% stop loss for options
                    # Check DTE (days to expiry) from symbol name
                    dte_warn = ""
                    try:
                        # Extract date from symbol like US.SLV260529C74000
                        date_part = sym.split(".")[1][:9]  # SLV260529
                        ticker_len = 3
                        for c in date_part:
                            if c.isdigit():
                                break
                            ticker_len += 1
                        exp_str = "20" + date_part[ticker_len - 3:][:6]
                        exp_date = datetime.strptime(exp_str, "%Y%m%d")
                        dte = (exp_date - datetime.now()).days
                        if dte <= 5:
                            dte_warn = f" [!!{dte}DTE EXPIRING]"
                            opt_sl = 0.30  # tighter SL near expiry
                        elif dte <= 10:
                            dte_warn = f" [{dte}DTE]"
                    except Exception:
                        dte_warn = ""

                    action = None
                    reason = ""
                    if pnl_ratio >= opt_tp:
                        action = "TAKE_PROFIT"
                        reason = f"Option +{pnl_ratio:.1%} >= +{opt_tp:.0%}"
                    elif pnl_ratio <= -opt_sl:
                        action = "STOP_LOSS"
                        reason = f"Option {pnl_ratio:.1%} <= -{opt_sl:.0%}"

                    tag = dte_warn
                    if not action:
                        if pnl_ratio >= opt_tp * 0.7:
                            tag += " [!TP near]"
                        elif pnl_ratio <= -opt_sl * 0.7:
                            tag += " [!SL near]"
                else:
                    action = None
                    reason = ""
                    if pnl_ratio >= take_profit:
                        action = "TAKE_PROFIT"
                        reason = f"+{pnl_ratio:.1%} >= +{take_profit:.0%}"
                    elif pnl_ratio <= -stop_loss:
                        action = "STOP_LOSS"
                        reason = f"{pnl_ratio:.1%} <= -{stop_loss:.0%}"

                    tag = ""
                    if pnl_ratio >= take_profit * 0.8:
                        tag = " [!TP near]"
                    elif pnl_ratio <= -stop_loss * 0.8:
                        tag = " [!SL near]"

                label = "OPT" if is_option else "ETF"
                if scan_count % 5 == 1 or action or tag:
                    print(f"  [{_now()}] [{label}] {sym:28s} x{qty:2d} | "
                          f"cost=${cost:.2f} now=${cur_price:.2f} | "
                          f"PnL={pnl_ratio:+.1%} (${pnl_val:+.2f}){tag}")

                if action:
                    triggered.append({
                        "symbol": sym, "qty": qty, "action": action,
                        "reason": reason, "pnl_ratio": pnl_ratio,
                        "pnl_val": pnl_val, "price": cur_price, "cost": cost,
                        "is_option": is_option,
                    })

            # Process triggers
            for t in triggered:
                label = "TAKE PROFIT" if t["action"] == "TAKE_PROFIT" else "STOP LOSS"
                is_opt = t.get("is_option", False)
                type_label = "OPTION" if is_opt else "ETF"
                print(f"\n  *** {label} ({type_label}): {t['symbol']} | {t['reason']} ***")

                sell_msg = (f"<b>{label} ({type_label})</b> {t['symbol']}\n"
                            f"Cost: ${t['cost']:.2f} -> ${t['price']:.2f}\n"
                            f"PnL: {t['pnl_ratio']:+.1%} (${t['pnl_val']:+.2f})\n"
                            f"Reason: {t['reason']}")

                sold = execute_sell(trade_ctx, t["symbol"], t["qty"],
                                    t["price"], dry_run,
                                    reason=t["reason"])
                if not sold:
                    notify(f"<b>SELL FAILED</b> {t['symbol']}")
                    continue

                store.remove_position(t["symbol"], t["price"], t["reason"])
                notify(sell_msg)

                # Options: no replacement scan, just pocket the profit/cut loss
                if is_opt:
                    print(f"  [{_now()}] Option closed, no replacement needed")
                    continue

                freed_usd = t["qty"] * t["price"]
                time.sleep(3)

                # ETF: scan for replacement
                print(f"  [{_now()}] Scanning for ETF replacement...")
                held = {p["symbol"] for p in store.get_positions()}
                candidates = scan_momentum(quote_ctx, held)

                if candidates:
                    best = candidates[0]
                    print(f"  [{_now()}] Best: {best['symbol']} "
                          f"(mom={best['momentum']:+.1%}, ${best['price']:.2f})")

                    ok, buy_qty, buy_price = execute_buy(
                        trade_ctx, quote_ctx, best["symbol"],
                        freed_usd, dry_run,
                    )
                    if ok and buy_qty > 0:
                        store.add_position(best["symbol"], buy_qty, buy_price)
                        notify(f"<b>ROTATE IN</b> {best['symbol']}\n"
                               f"Qty: {buy_qty} @ ${buy_price:.2f}\n"
                               f"Momentum: {best['momentum']:+.1%}\n"
                               f"Replaced: {t['symbol']}")
                    else:
                        print(f"  [{_now()}] Buy failed, cash idle")
                else:
                    print(f"  [{_now()}] No qualified replacement, cash idle")
                    notify(f"No replacement for {t['symbol']}, cash idle")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n  Monitor stopped by user")
    finally:
        quote_ctx.close()
        trade_ctx.close()
        notify("<b>ETF Monitor Stopped</b>")


def init_from_broker():
    """Sync current broker positions into the persistent DB."""
    from futu import OpenSecTradeContext, TrdMarket
    store = PositionStore()
    trade_ctx = OpenSecTradeContext(host="127.0.0.1", port=11111,
                                    filter_trdmarket=TrdMarket.US)
    try:
        store.sync_from_broker(trade_ctx)
        print("Positions synced from broker:")
        for p in store.get_positions():
            print(f"  {p['symbol']:10s} x{p['qty']:2d} @ ${p['cost_price']:.2f}")
    finally:
        trade_ctx.close()


def show_status():
    """Show DB contents without connecting to broker."""
    store = PositionStore()
    positions = store.get_positions()
    print(f"\n  Tracked positions ({len(positions)}):")
    for p in positions:
        print(f"    {p['symbol']:10s} x{p['qty']:2d} @ ${p['cost_price']:.2f} "
              f"(since {p['bought_at'][:10]})")
    trades = store.get_trade_log(10)
    if trades:
        print(f"\n  Recent trades:")
        for t in trades:
            print(f"    {t['timestamp'][:16]} {t['action']:4s} {t['symbol']:10s} "
                  f"x{t['qty']} @ ${t['price']:.2f} "
                  f"pnl=${t['pnl']:+.2f} {t['reason']}")


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="ETF position monitor (persistent)")
    parser.add_argument("--dry-run", action="store_true", help="Watch only")
    parser.add_argument("--tp", type=float, default=0.15, help="Take profit (default 0.15)")
    parser.add_argument("--sl", type=float, default=0.08, help="Stop loss (default 0.08)")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval seconds")
    parser.add_argument("--init", action="store_true", help="Sync broker positions to DB")
    parser.add_argument("--status", action="store_true", help="Show DB status")
    args = parser.parse_args()

    if args.init:
        init_from_broker()
    elif args.status:
        show_status()
    else:
        run_monitor(take_profit=args.tp, stop_loss=args.sl,
                    interval=args.interval, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
