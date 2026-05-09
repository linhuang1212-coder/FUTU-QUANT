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


def _parse_option_code(code: str) -> dict:
    """Parse Futu option code into components.

    E.g. 'US.IWM260522P268000' -> {underlying: 'US.IWM', option_type: 'PUT',
    strike: 268.0, expiry: '2026-05-22'}
    """
    result = {"underlying": "", "option_type": "", "strike": 0.0, "expiry": ""}
    if "." not in code:
        return result
    parts = code.split(".")
    if len(parts) < 2:
        return result

    ticker_part = parts[1]
    alpha = ""
    rest = ""
    for i, ch in enumerate(ticker_part):
        if ch.isalpha():
            alpha += ch
        else:
            rest = ticker_part[i:]
            break
    result["underlying"] = f"US.{alpha}" if alpha else ""

    # rest = "260522P268000" -> date=260522, type=P, strike_raw=268000
    pc_idx = -1
    for i, ch in enumerate(rest):
        if ch in ("P", "C"):
            pc_idx = i
            break
    if pc_idx < 0:
        return result

    date_str = rest[:pc_idx]       # "260522"
    opt_char = rest[pc_idx]        # "P" or "C"
    strike_raw = rest[pc_idx + 1:]  # "268000"

    result["option_type"] = "PUT" if opt_char == "P" else "CALL"
    try:
        result["strike"] = int(strike_raw) / 1000.0
    except ValueError:
        pass
    if len(date_str) == 6:
        result["expiry"] = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"

    return result


def _get_real_positions(trade_ctx, trade_env: str) -> list[dict]:
    """Query Futu for real OPTION positions only.

    Strictly filters out stock/ETF holdings by requiring valid
    option_type + strike + expiry from the code parser.
    """
    from futu import RET_OK, TrdEnv
    env = TrdEnv.REAL if trade_env == "REAL" else TrdEnv.SIMULATE
    ret, positions = trade_ctx.position_list_query(trd_env=env)
    if ret != RET_OK or positions is None or positions.empty:
        return []

    result = []
    for _, pos in positions.iterrows():
        qty = int(pos.get("qty", 0))
        if qty == 0:
            continue
        code = str(pos.get("code", ""))

        parsed = _parse_option_code(code)
        # Must have all three: strike > 0, valid expiry, valid option_type
        if not (parsed["strike"] > 0 and parsed["expiry"]
                and parsed["option_type"] in ("PUT", "CALL")):
            logger.info(f"[STARTUP] 跳过非期权持仓: {code} qty={qty}")
            continue

        cost = float(pos.get("cost_price", 0))
        mkt_val = abs(float(pos.get("market_val", 0)))
        underlying = parsed["underlying"]

        if qty < 0:
            strategy = "credit_spread"  # short leg of a spread
            premium = cost * abs(qty) * 100
        else:
            strategy = "credit_spread"  # long leg of a spread
            premium = cost * qty * 100
        max_loss = mkt_val

        logger.info(f"[STARTUP] 期权持仓: {code} qty={qty} cost=${cost:.2f} "
                    f"-> {strategy} on {underlying}")
        result.append({
            "code": code,
            "underlying": underlying,
            "strategy": strategy,
            "premium": premium,
            "max_loss": max_loss,
            "qty": qty,
            "cost": cost,
            "option_type": parsed["option_type"],
            "strike": parsed["strike"],
            "expiry": parsed["expiry"],
        })
    return result


def _reconstruct_trades(existing_positions: list[dict], config: dict) -> list[OptionTrade]:
    """Reconstruct OptionTrade objects from real broker positions.

    Groups legs by underlying to reconstruct multi-leg trades (e.g. credit spreads).
    Single-leg positions become standalone trades (e.g. wheel CSP).
    """
    from options.order import OptionLeg

    # Group by (underlying, expiry) to correctly separate spreads
    # with different expiry dates on the same underlying
    by_key: dict[tuple[str, str], list[dict]] = {}
    for pos in existing_positions:
        und = pos["underlying"]
        expiry = pos.get("expiry", "unknown")
        key = (und, expiry)
        by_key.setdefault(key, []).append(pos)

    cs_cfg = config.get("credit_spread", {})
    wheel_cfg = config.get("wheel", {})

    trades = []
    for (und, expiry), legs_data in by_key.items():
        has_buy = any(p["qty"] > 0 for p in legs_data)
        has_sell = any(p["qty"] < 0 for p in legs_data)

        if has_buy and has_sell:
            option_legs = []
            for p in legs_data:
                direction = "BUY" if p["qty"] > 0 else "SELL"
                leg = OptionLeg(
                    code=p["code"], underlying=und,
                    direction=direction, qty=abs(p["qty"]),
                    option_type=p["option_type"], strike=p["strike"],
                    expiry=p["expiry"], price=p["cost"],
                    fill_price=p["cost"],
                )
                option_legs.append(leg)

            sell_legs = [l for l in option_legs if l.direction == "SELL"]
            buy_legs = [l for l in option_legs if l.direction == "BUY"]
            if sell_legs and buy_legs:
                width = abs(sell_legs[0].strike - buy_legs[0].strike)
                net_credit = sell_legs[0].fill_price - buy_legs[0].fill_price
                max_loss = (width - max(net_credit, 0)) * 100
            else:
                max_loss = sum(p["max_loss"] for p in legs_data)

            trade = OptionTrade(
                strategy="credit_spread", underlying=und, legs=option_legs,
                max_loss=max_loss,
                target_pnl=max_loss * cs_cfg.get("profit_take_pct", 0.50),
                stop_loss_pct=cs_cfg.get("stop_loss_pct", 2.00),
                take_profit_pct=cs_cfg.get("profit_take_pct", 0.50),
                status="open",
                open_timestamp="restored",
            )
            trades.append(trade)
            logger.info(f"[RECONSTRUCT] Credit Spread on {und} exp={expiry}: "
                        f"{len(option_legs)} legs, max_loss=${max_loss:.0f}")
        else:
            # Single-leg trades (wheel CSP, orphaned spread leg, or standalone buy)
            for p in legs_data:
                direction = "BUY" if p["qty"] > 0 else "SELL"
                leg = OptionLeg(
                    code=p["code"], underlying=und,
                    direction=direction, qty=abs(p["qty"]),
                    option_type=p["option_type"], strike=p["strike"],
                    expiry=p["expiry"], price=p["cost"],
                    fill_price=p["cost"],
                )

                if direction == "SELL":
                    strategy = "wheel_csp"
                    sl_pct = 2.00
                    tp_pct = wheel_cfg.get("profit_take_pct", 0.50)
                else:
                    # Long option only — likely an orphaned spread leg or
                    # standalone buy; use conservative SL to exit quickly.
                    strategy = "orphan_long"
                    sl_pct = 0.50
                    tp_pct = 0.50

                trade = OptionTrade(
                    strategy=strategy, underlying=und, legs=[leg],
                    max_loss=p["max_loss"],
                    target_pnl=p["premium"] * tp_pct,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                    status="open",
                    open_timestamp=datetime.now().isoformat(),
                )
                trades.append(trade)
                logger.info(f"[RECONSTRUCT] {strategy} on {und}: "
                            f"{leg.code} qty={leg.qty}, max_loss=${p['max_loss']:.0f}, "
                            f"SL={sl_pct:.0%} TP={tp_pct:.0%}")

    return trades


def _monitor_positions(trader, trade_ctx, quote_ctx, trade_env: str, notifier,
                       emergency_loss_pct: float = 3.00, config: dict = None,
                       risk_mgr=None):
    """Monitor positions with multi-layer risk protection.

    Three layers of protection:
    1. trader.monitor_open_trades() — in-memory SL/TP enforcement
    2. Daily P&L circuit breaker — pause new trades if daily loss exceeds limit
    3. Emergency broker-level check — single-position loss alert
    """
    from futu import RET_OK, TrdEnv
    env = TrdEnv.REAL if trade_env == "REAL" else TrdEnv.SIMULATE

    cs_cfg = config if config else {}
    max_hold = int(cs_cfg.get("credit_spread", {}).get("max_holding_days", 45))

    risk_cfg = cs_cfg.get("risk", {})
    max_daily_loss = float(risk_cfg.get("max_daily_loss", 300))
    check_interval = 60  # seconds between checks
    _daily_loss_alerted = False
    _severe_loss_alerted = False

    # Factor library market timing check
    market_state = "NEUTRAL"
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
                market_state = timing["market_state"]
                if market_state == "BEARISH":
                    check_interval = 30
                    logger.warning("[MONITOR] 因子库熊市信号 → 加密监控 (30s)")
    except Exception:
        pass

    while True:
        try:
            # Layer 1: precise SL/TP + expiry guard
            if trader.get_open_trades():
                trader.monitor_open_trades(max_holding_days=max_hold)

            # Layer 2+3: broker-level monitoring
            ret, positions = trade_ctx.position_list_query(trd_env=env)
            if ret != RET_OK or positions is None or positions.empty:
                logger.info("[MONITOR] 所有持仓已平仓，退出监控")
                break

            active = positions[positions["qty"] != 0]
            if active.empty:
                logger.info("[MONITOR] 所有持仓已平仓，退出监控")
                break

            tracked_codes = set()
            for t in trader.get_open_trades():
                for leg in t.legs:
                    tracked_codes.add(leg.code)

            total_pnl = 0.0
            worst_position = None
            worst_pnl = 0.0

            for _, pos in active.iterrows():
                code = str(pos.get("code", ""))
                qty = int(pos.get("qty", 0))
                pnl = float(pos.get("pl_val", 0))
                cost = float(pos.get("cost_price", 0))
                pnl_pct = float(pos.get("pl_ratio", 0)) * 100
                total_pnl += pnl

                if pnl < worst_pnl:
                    worst_pnl = pnl
                    worst_position = code

                # Emergency alert: untracked positions losing big
                if code not in tracked_codes and cost > 0 and pnl < 0:
                    loss_ratio = abs(pnl) / (cost * abs(qty) * 100)
                    if loss_ratio >= emergency_loss_pct:
                        logger.warning(f"[EMERGENCY SL] {code} 亏损 {pnl_pct:+.1f}% "
                                       f"(${pnl:+.2f}) 超过紧急止损线")
                        if notifier:
                            notifier.send_sync(
                                f"<b>紧急止损警报</b>\n"
                                f"持仓: <code>{code}</code>\n"
                                f"亏损: ${pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                                f"原因: 超过紧急止损线 ({emergency_loss_pct:.0%})\n"
                                f"<b>请手动检查是否需要平仓</b>"
                            )

                logger.debug(f"[MONITOR] {code} qty={qty} pnl=${pnl:+.2f} ({pnl_pct:+.1f}%)")

            # ── Layer 2: Daily P&L circuit breaker ──
            if total_pnl <= -max_daily_loss and not _daily_loss_alerted:
                _daily_loss_alerted = True
                logger.warning(f"[CIRCUIT BREAKER] 日亏触及上限: "
                               f"${total_pnl:+.2f} <= -${max_daily_loss:.0f}")
                if notifier:
                    notifier.send_sync(
                        f"<b>🚨 日亏触及上限</b>\n"
                        f"当前浮盈亏: ${total_pnl:+.2f}\n"
                        f"日亏上限: -${max_daily_loss:.0f}\n"
                        f"最差持仓: <code>{worst_position}</code> (${worst_pnl:+.2f})\n"
                        f"<b>系统已暂停新开仓</b>"
                    )
                if risk_mgr:
                    risk_mgr._daily_pnl = total_pnl

            # Severe loss alert (50% of monthly drawdown limit)
            monthly_dd = float(risk_cfg.get("max_monthly_drawdown", 600))
            if total_pnl <= -(monthly_dd * 0.5) and not _severe_loss_alerted:
                _severe_loss_alerted = True
                logger.warning(f"[SEVERE] 已触及月度回撤50%警戒线: ${total_pnl:+.2f}")
                if notifier:
                    notifier.send_sync(
                        f"<b>⚠️ 月度回撤50%警戒</b>\n"
                        f"当前浮盈亏: ${total_pnl:+.2f}\n"
                        f"月度回撤上限: -${monthly_dd:.0f}\n"
                        f"<b>建议检查所有持仓</b>"
                    )

            logger.info(f"[MONITOR] {len(active)} 个持仓 | "
                        f"总浮盈亏: ${total_pnl:+.2f} | "
                        f"市场状态: {market_state}")

        except Exception as e:
            logger.error(f"[MONITOR] 监控异常: {e}")
            if notifier:
                notifier.send_sync(f"<b>监控异常</b>\n{str(e)[:200]}")

        time.sleep(check_interval)


def load_options_config() -> dict:
    cfg_path = get_project_root() / "config" / "options.yaml"
    cfg = load_yaml(str(cfg_path))
    return cfg.get("options", {})


def run_live(strategies: list[str], dry_run: bool, config: dict):
    """Run live/simulate option strategies."""
    from futu import OpenQuoteContext, OpenSecTradeContext, TrdEnv

    # Global trade lock
    from utils.trade_lock import acquire_trade_lock
    acquire_trade_lock("run_options")

    trade_env = config.get("trade_env", "SIMULATE")
    logger.info(f"启动期权交易系统 | 环境={trade_env} | dry_run={dry_run} | 策略={strategies}")

    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    trade_ctx = OpenSecTradeContext(host="127.0.0.1", port=11111)
    store = TradeStore()

    from options.chain import OptionChainFetcher
    from options.trader import OptionsTrader
    from options.scanner import VolatilityScanner

    from risk.pdt_guard import PdtGuard
    from notification.telegram_bot import TelegramNotifier

    chain = OptionChainFetcher(quote_ctx)
    pdt_guard = PdtGuard(max_day_trades=3, rolling_window_days=5, trade_store=store)
    risk_mgr = OptionsRiskManager(config, pdt_guard=pdt_guard)
    fmp_key = config.get("fmp_api_key", "") or os.environ.get("FMP_API_KEY", "")

    # Telegram
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    notifier = TelegramNotifier(tg_token, tg_chat, enabled=bool(tg_token and tg_chat))

    trader = OptionsTrader(trade_ctx, quote_ctx, risk_mgr,
                           trade_env=trade_env, dry_run=dry_run,
                           fmp_api_key=fmp_key, notifier=notifier,
                           trade_store=store)
    scanner = VolatilityScanner(quote_ctx)

    # Check real positions → reconstruct trades → inject into trader + risk_mgr
    existing_positions = _get_real_positions(trade_ctx, trade_env)
    if existing_positions:
        logger.info(f"[STARTUP] 发现 {len(existing_positions)} 个实盘持仓")

        # Inject into risk manager
        for pos in existing_positions:
            risk_mgr.on_trade_open(
                pos["strategy"], pos["premium"],
                underlying=pos["underlying"], max_loss=pos["max_loss"],
            )

        # Reconstruct OptionTrade objects with proper SL/TP settings
        restored_trades = _reconstruct_trades(existing_positions, config)
        for trade in restored_trades:
            trade.dry_run = dry_run
            trader._open_trades.append(trade)
        logger.info(f"[STARTUP] 重建 {len(restored_trades)} 个交易对象 "
                    f"(含止损止盈参数)，注入监控系统")

    mode = "实盘" if not dry_run else "模拟"
    notifier.send_sync(
        f"<b>FUTU-QUANT 启动</b>\n"
        f"模式: {mode} | 环境: {trade_env}\n"
        f"策略: {', '.join(strategies)}\n"
        f"已有持仓: {len(existing_positions)} 个"
    )

    try:
        # ── Strategy A: 0DTE ORB (已禁用) ──
        if "orb" in strategies or "all" in strategies:
            orb_per = config.get("risk", {}).get("per_strategy", {}).get("orb", {})
            if not orb_per.get("enabled", False):
                logger.info("[ORB] 策略已禁用，跳过")
            else:
                _run_orb(chain, trader, store, config.get("orb", {}), dry_run, config)

        # ── Strategy B: Earnings Spread ──
        if "earnings" in strategies or "all" in strategies:
            _run_earnings(chain, trader, store, config.get("earnings", {}), dry_run)

        # ── Strategy C: Credit Spread ──
        if "credit_spread" in strategies or "all" in strategies:
            cs_per = config.get("risk", {}).get("per_strategy", {}).get("credit_spread", {})
            if not cs_per.get("enabled", True):
                logger.info("[CREDIT] Credit Spread 已禁用，跳过")
            else:
                _run_credit_spread(chain, trader, scanner, store,
                                   config.get("credit_spread", {}), config, dry_run,
                                   quote_ctx)

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

        # Monitor loop — run if we have in-memory trades OR real positions
        has_real_positions = bool(existing_positions)
        has_open_trades = bool(trader.get_open_trades())
        if has_real_positions or has_open_trades:
            logger.info(f"进入持仓监控循环 (内存={len(trader.get_open_trades())} "
                        f"实盘={len(existing_positions)}) Ctrl+C 退出...")
            _monitor_positions(trader, trade_ctx, quote_ctx, trade_env, notifier,
                               config=config, risk_mgr=risk_mgr)
        else:
            logger.info("无持仓，跳过监控循环")

    except KeyboardInterrupt:
        logger.info("用户中断，退出...")
    finally:
        _save_open_trades(trader, store, dry_run)
        open_count = len(trader.get_open_trades())
        notifier.send_sync(f"<b>FUTU-QUANT 已停止</b>\n未平仓位: {open_count} 个")
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
    from data.earnings_calendar import get_upcoming_earnings
    from options.scanner import VolatilityScanner

    strat = EarningsSpreadStrategy(config, chain)
    symbols = config.get("scan_symbols", [])
    days_before = config.get("days_before", 7)
    fmp_key = os.environ.get("FMP_API_KEY", "")

    logger.info(f"[EARNINGS] 扫描 {len(symbols)} 只标的, 未来 {days_before} 天财报...")

    upcoming = get_upcoming_earnings(symbols, days_ahead=days_before,
                                     fmp_api_key=fmp_key)
    if not upcoming:
        logger.info("[EARNINGS] 未发现即将财报的标的")
        return

    logger.info(f"[EARNINGS] 发现 {len(upcoming)} 只即将财报:")
    for u in upcoming:
        logger.info(f"  {u['symbol']} -> {u['earnings_date']} ({u['source']})")

    scanner = VolatilityScanner(chain._ctx)
    ivr_hits = scanner.scan_ivr([u["symbol"] for u in upcoming], min_ivr=0)
    ivr_map = {h["symbol"]: h["ivr"] for h in ivr_hits}

    for entry in upcoming:
        sym = entry["symbol"]
        earn_date = entry["earnings_date"]
        ivr = ivr_map.get(sym, 50.0)

        # Simple uptrend check via recent price vs SMA
        is_uptrend = True
        try:
            from futu import RET_OK, KLType
            ret, data = chain._ctx.get_cur_kline(sym, 50, KLType.K_DAY)
            if ret == RET_OK and data is not None and len(data) >= 20:
                closes = data["close"].values
                sma20 = float(closes[-20:].mean())
                is_uptrend = closes[-1] > sma20
        except Exception:
            pass

        logger.info(f"[EARNINGS] {sym} 财报={earn_date} IVR={ivr:.0f} "
                    f"趋势={'上' if is_uptrend else '下'}")

        trade = strat.evaluate(sym, earn_date, ivr, is_uptrend)
        if trade:
            trade.dry_run = dry_run
            if trader.open_trade(trade):
                trade.trade_id = store.log_option_trade(
                    strategy=trade.strategy, underlying=trade.underlying,
                    legs_json=trade.legs_json(), max_loss=trade.max_loss,
                    target_pnl=trade.target_pnl, dry_run=dry_run,
                )


def _get_factor_library_candidates(model: str = "credit_spread",
                                   top_n: int = 30) -> list[str]:
    """Load pre-computed factor library scores and return top candidate symbols.

    Falls back to empty list if factor library data is unavailable.
    """
    try:
        from factor_library.storage import load_factors
        from factor_library.search import build_factor_matrix
        from factor_library.screener import score_stocks, risk_filter

        categories = ["technical", "risk", "volatility", "liquidity", "fundamental"]
        factor_dfs = {}
        for cat in categories:
            df = load_factors(cat)
            if not df.empty:
                factor_dfs[cat] = df

        if not factor_dfs:
            return []

        matrix = build_factor_matrix(factor_dfs)
        if matrix.empty:
            return []

        filtered = risk_filter(matrix, top_pct=0.8)
        if filtered.empty:
            filtered = matrix

        # Exclude illiquid stocks: require top-40% turnover for option spreads
        if "TURNOVER" in filtered.columns:
            turnover_thresh = filtered["TURNOVER"].quantile(0.6)
            liquid = filtered[filtered["TURNOVER"] >= turnover_thresh]
            if len(liquid) >= 50:
                filtered = liquid

        results = score_stocks(filtered, model=model, top_n=top_n)
        if results.empty:
            return []

        symbols = [f"US.{s}" for s in results["symbol"].tolist()]
        logger.info(f"[FACTOR-LIB] {model} 模型筛选 {len(symbols)} 只候选标的")
        return symbols
    except Exception as e:
        logger.debug(f"[FACTOR-LIB] Factor library unavailable ({e})")
        return []


def _rank_ivr_hits_by_factor(ivr_hits: list[dict], quote_ctx=None) -> list[dict]:
    """Re-rank IVR hits using factor library scores (preferred) or live factor calc."""
    if len(ivr_hits) < 2:
        return ivr_hits

    # Try factor library first (pre-computed, fast)
    try:
        from factor_library.storage import load_factors
        from factor_library.search import build_factor_matrix
        from factor_library.screener import score_stocks

        categories = ["technical", "risk", "volatility", "liquidity"]
        factor_dfs = {}
        for cat in categories:
            df = load_factors(cat)
            if not df.empty:
                factor_dfs[cat] = df

        if factor_dfs:
            matrix = build_factor_matrix(factor_dfs)
            if not matrix.empty:
                cs_results = score_stocks(matrix, model="credit_spread",
                                          top_n=len(matrix))
                if not cs_results.empty:
                    score_map = dict(zip(cs_results["symbol"],
                                         cs_results["score"]))
                    for hit in ivr_hits:
                        ticker = hit["symbol"].replace("US.", "")
                        hit["factor_score"] = round(
                            score_map.get(ticker, 0.0), 4)

                    ivr_hits.sort(key=lambda h: h.get("factor_score", 0),
                                  reverse=True)
                    for h in ivr_hits:
                        logger.info(
                            f"[FACTOR-LIB] {h['symbol']} IVR={h['ivr']:.0f} "
                            f"factor_score={h.get('factor_score', 0):.4f}")
                    return ivr_hits
    except Exception as e:
        logger.debug(f"[FACTOR-LIB] Ranking via factor library failed ({e})")

    # Fallback: live factor calc
    try:
        from factor.data_provider import FactorDataProvider
        from factor.technical import calc_momentum, calc_volatility
        from factor.processor import cross_sectional_rank

        symbols = [h["symbol"] for h in ivr_hits]
        provider = FactorDataProvider(quote_ctx=quote_ctx)
        prices, volumes = provider.get_daily_panel(symbols, years=1)
        if prices.empty or len(prices) < 63:
            return ivr_hits

        returns = prices.pct_change()
        mom3m = cross_sectional_rank(calc_momentum(prices, 63))
        vol60 = cross_sectional_rank(calc_volatility(returns, 60))

        latest_mom = mom3m.iloc[-1] if not mom3m.empty else pd.Series(dtype=float)
        latest_vol = vol60.iloc[-1] if not vol60.empty else pd.Series(dtype=float)

        from data.downloader import _normalize_symbol
        for hit in ivr_hits:
            norm = _normalize_symbol(hit["symbol"])
            m = latest_mom.get(norm, 0.5)
            v = latest_vol.get(norm, 0.5)
            ivr_rank = hit["ivr"] / 100.0
            hit["factor_score"] = round(0.4 * m + 0.3 * v + 0.3 * ivr_rank, 4)

        ivr_hits.sort(key=lambda h: h.get("factor_score", 0), reverse=True)
        for h in ivr_hits:
            logger.info(f"[FACTOR] {h['symbol']} IVR={h['ivr']:.0f} "
                        f"factor_score={h.get('factor_score', 0):.3f}")
        return ivr_hits
    except Exception as e:
        logger.debug(f"[FACTOR] Factor ranking failed ({e}), using IVR order")
        return ivr_hits


def _run_credit_spread(chain, trader, scanner, store, config, full_config, dry_run,
                       quote_ctx=None):
    from options.strategies.credit_spread import CreditSpreadStrategy
    from options.strategies.direction import DirectionAnalyzer

    yaml_symbols = config.get("scan_symbols", [])
    min_ivr = config.get("min_ivr", 60)
    max_pos = config.get("max_positions", 3)
    dte_list = config.get("target_dte_list", [21])

    per_strat = full_config.get("risk", {}).get("per_strategy", {}).get("credit_spread", {})
    min_credit = float(per_strat.get("min_credit_received", 0.15))

    dir_config = config.get("direction", {})
    dir_enabled = dir_config.get("enabled", True)

    # Factor library integration: merge factor-screened candidates with YAML list
    factor_candidates = _get_factor_library_candidates(
        model="credit_spread", top_n=20)
    if factor_candidates:
        seen = set(yaml_symbols)
        for fc in factor_candidates:
            if fc not in seen:
                yaml_symbols.append(fc)
                seen.add(fc)
        logger.info(f"[CREDIT] 因子库补充 {len(factor_candidates)} 只候选, "
                    f"合并后共 {len(yaml_symbols)} 只")
    symbols = yaml_symbols

    logger.info(f"[CREDIT] 扫描 {len(symbols)} 个标的 x {len(dte_list)} 到期周期 "
                f"(min IVR={min_ivr}, min credit=${min_credit:.2f}, "
                f"width=${config.get('spread_width', 5)}, 方向判断={'开' if dir_enabled else '关'})...")
    ivr_hits = scanner.scan_ivr(symbols, min_ivr=min_ivr)

    # Multi-factor ranking: prioritize IVR hits by momentum + volatility + IVR
    ivr_hits = _rank_ivr_hits_by_factor(ivr_hits, quote_ctx=quote_ctx)

    strat = CreditSpreadStrategy(config, chain, min_credit=min_credit)
    ctx = quote_ctx or getattr(chain, '_ctx', None)
    direction_analyzer = DirectionAnalyzer(ctx, dir_config) if dir_enabled and ctx else None

    # Count existing credit spread positions from trader's open trades
    existing_cs = sum(1 for t in trader.get_open_trades()
                      if getattr(t, 'strategy', '') == 'credit_spread')
    opened = existing_cs
    used_symbols = set()
    if existing_cs > 0:
        logger.info(f"[CREDIT] 已有 {existing_cs} 组 Credit Spread 持仓 "
                    f"(上限={max_pos})，扣除后可开 {max(0, max_pos - existing_cs)} 组")

    for target_dte in dte_list:
        if opened >= max_pos:
            break
        for hit_idx, hit in enumerate(ivr_hits):
            if opened >= max_pos:
                break
            sym = hit["symbol"]
            cycle_key = f"{sym}_{target_dte}"
            if cycle_key in used_symbols:
                continue

            if hit_idx > 0:
                time.sleep(3)

            # Direction analysis
            direction = "BULL"
            dir_details = ""
            if direction_analyzer:
                signal = direction_analyzer.analyze(sym)
                if signal is None:
                    logger.info(f"[CREDIT] {sym} 方向分析失败，跳过")
                    continue
                if signal.direction == "NEUTRAL":
                    logger.info(f"[CREDIT] {sym} 方向不明 (评分={signal.score:+.1f})，跳过")
                    continue
                direction = signal.direction
                dir_details = signal.details
                spread_type = "Bull Put" if direction == "BULL" else "Bear Call"
                logger.info(f"[CREDIT] {sym} 方向={direction} ({spread_type}) "
                            f"评分={signal.score:+.1f}")

            trade = strat.evaluate(sym, hit["ivr"], min_ivr,
                                   target_dte=target_dte, direction=direction)
            if trade:
                trade.dry_run = dry_run
                trade.direction_details = dir_details
                if trader.open_trade(trade):
                    trade.trade_id = store.log_option_trade(
                        strategy=trade.strategy, underlying=trade.underlying,
                        legs_json=trade.legs_json(), max_loss=trade.max_loss,
                        target_pnl=trade.target_pnl, dry_run=dry_run,
                    )
                    opened += 1
                    used_symbols.add(cycle_key)

    logger.info(f"[CREDIT] 开仓 {opened} 组 Credit Spread (across {len(dte_list)} cycles)")


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
    """Fallback: persist any closed trades that weren't saved during close_trade."""
    for trade in trader.get_open_trades():
        if trade.trade_id and trade.status == "closed":
            store.close_option_trade(trade.trade_id, trade.realized_pnl, trade.close_reason)
    for trade in getattr(trader, '_closed_trades', []):
        if trade.trade_id and trade.status == "closed":
            try:
                store.close_option_trade(trade.trade_id, trade.realized_pnl, trade.close_reason)
            except Exception:
                pass


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
