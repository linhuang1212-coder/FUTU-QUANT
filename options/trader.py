from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Optional
from options.order import OptionLeg, OptionTrade
from options.risk import OptionsRiskManager, EarningsGuard
from utils.logger import setup_logger

logger = setup_logger("options.trader")


class OptionsTrader:
    """Execute option trades via Futu OpenAPI."""

    def __init__(self, trade_ctx, quote_ctx, risk_mgr: OptionsRiskManager,
                 trade_env: str = "SIMULATE", dry_run: bool = False,
                 fmp_api_key: str = "", notifier=None, trade_store=None):
        self._trade_ctx = trade_ctx
        self._quote_ctx = quote_ctx
        self.risk = risk_mgr
        self.earnings_guard = EarningsGuard(quote_ctx, fmp_api_key=fmp_api_key)
        self.trade_env = trade_env
        self.dry_run = dry_run
        self._open_trades: list[OptionTrade] = []
        self._notifier = notifier
        self._store = trade_store

    def open_trade(self, trade: OptionTrade) -> bool:
        """Open a multi-leg option trade."""
        # Earnings check for individual stocks
        if trade.legs:
            expiry = trade.legs[0].expiry
            safe, reason = self.earnings_guard.check(trade.underlying, expiry)
            if not safe:
                logger.warning(f"[EARNINGS BLOCK] {trade.strategy}: {reason}")
                self._notify(f"<b>财报拦截</b> {trade.strategy}\n标的: {trade.underlying}\n原因: {reason}")
                return False

        # Reject trades with any zero-price legs
        for leg in trade.legs:
            if leg.price <= 0:
                logger.warning(f"[PRICE BLOCK] {leg.code} price=${leg.price:.2f} <= 0, skip")
                return False

        premium = abs(trade.net_premium()) if any(l.price > 0 for l in trade.legs) else trade.max_loss
        ok, reason = self.risk.can_open_trade(
            trade.strategy, premium,
            underlying=trade.underlying, max_loss=trade.max_loss,
        )
        if not ok:
            logger.warning(f"[RISK BLOCK] {trade.strategy}: {reason}")
            return False

        # Pre-trade buying power check (live only)
        if not self.dry_run:
            buying_power = self._check_buying_power()
            if buying_power is not None and buying_power < trade.max_loss * 1.2:
                logger.warning(f"[BUYING POWER] ${buying_power:.0f} < "
                               f"need ${trade.max_loss * 1.2:.0f}, skip {trade.underlying}")
                return False

        logger.info(f"[OPEN] {trade.strategy} on {trade.underlying} "
                    f"({len(trade.legs)} legs, max_loss=${trade.max_loss:.0f})")

        if self.dry_run:
            for leg in trade.legs:
                leg.fill_price = leg.price
                leg.order_id = "DRY_RUN"
                logger.info(f"  [DRY] {leg.direction} {leg.qty}x {leg.code} @ ${leg.price:.2f}")
            trade.status = "open"
            trade.open_timestamp = __import__("datetime").datetime.now().isoformat()
            self.risk.on_trade_open(trade.strategy, premium,
                                    underlying=trade.underlying,
                                    max_loss=trade.max_loss)
            self._open_trades.append(trade)
            self._notify_open(trade)
            return True

        all_ok = True
        filled_legs: list[OptionLeg] = []
        # SELL legs first: if SELL fails we abort with zero risk;
        # if BUY (protection) fails we rollback the SELL leg
        sorted_legs = sorted(trade.legs, key=lambda l: 0 if l.direction == "SELL" else 1)
        for leg in sorted_legs:
            success = self._place_leg(leg)
            if not success:
                all_ok = False
                logger.error(f"  [FAIL] {leg.direction} {leg.code}")
                break
            filled_legs.append(leg)
            time.sleep(0.5)

        if all_ok:
            trade.status = "open"
            trade.open_timestamp = __import__("datetime").datetime.now().isoformat()
            self.risk.on_trade_open(trade.strategy, premium,
                                    underlying=trade.underlying,
                                    max_loss=trade.max_loss)
            self._open_trades.append(trade)
            self._notify_open(trade)
        else:
            if filled_legs:
                self._rollback_filled_legs(filled_legs)
            self._notify(
                f"<b>下单失败</b> {trade.strategy}\n"
                f"标的: <code>{trade.underlying}</code>\n"
                f"原因: 购买力不足"
            )
        return all_ok

    def _check_buying_power(self) -> Optional[float]:
        """Query actual available funds from Futu account."""
        try:
            from futu import RET_OK, TrdEnv
            env = TrdEnv.REAL if self.trade_env == "REAL" else TrdEnv.SIMULATE
            ret, data = self._trade_ctx.accinfo_query(trd_env=env)
            if ret == RET_OK and data is not None and len(data) > 0:
                funds = float(data.iloc[0].get("available_funds", 0))
                logger.info(f"[BUYING POWER] ${funds:.0f}")
                return funds
        except Exception as e:
            logger.warning(f"[BUYING POWER] query failed: {e}")
        return None

    def _place_leg(self, leg: OptionLeg) -> bool:
        if leg.price <= 0:
            logger.error(f"  [FAIL] price ${leg.price:.2f} <= 0 for {leg.code}")
            return False

        from futu import RET_OK, TrdSide, OrderType as FutuOT, TrdEnv
        env = TrdEnv.REAL if self.trade_env == "REAL" else TrdEnv.SIMULATE
        side = TrdSide.BUY if leg.direction == "BUY" else TrdSide.SELL

        ret, data = self._trade_ctx.place_order(
            price=leg.price, qty=leg.qty, code=leg.code,
            trd_side=side, order_type=FutuOT.NORMAL, trd_env=env,
        )
        if ret == RET_OK and data is not None and len(data) > 0:
            leg.order_id = str(data.iloc[0].get("order_id", ""))
            leg.fill_price = leg.price
            logger.info(f"  [OK] {leg.direction} {leg.qty}x {leg.code} @ ${leg.price:.2f} "
                        f"order_id={leg.order_id}")
            return True
        logger.error(f"  [FAIL] place_order: {data}")
        return False

    def close_trade(self, trade: OptionTrade, reason: str = "") -> bool:
        """Close all legs of an open trade.

        On partial failure, removes the already-closed legs from the trade so
        the next monitor cycle only retries the remaining legs.
        """
        if trade.status != "open":
            return False

        logger.info(f"[CLOSE] {trade.strategy} on {trade.underlying}: {reason}")

        # Close short legs first (BUY back SELL-legs to release margin),
        # then close long legs (SELL BUY-legs)
        sorted_legs = sorted(trade.legs, key=lambda l: (0 if l.direction == "SELL" else 1))

        closed_legs: list[OptionLeg] = []
        failed = False
        for leg in sorted_legs:
            close_dir = "SELL" if leg.direction == "BUY" else "BUY"
            close_leg = OptionLeg(
                code=leg.code, underlying=leg.underlying,
                direction=close_dir, qty=leg.qty,
                option_type=leg.option_type, strike=leg.strike,
                expiry=leg.expiry, price=0,
            )
            if self._quote_ctx:
                from futu import RET_OK
                time.sleep(0.5)
                ret, snap = self._quote_ctx.get_market_snapshot([leg.code])
                if ret == RET_OK and snap is not None and len(snap) > 0:
                    if close_dir == "SELL":
                        close_leg.price = float(snap.iloc[0].get("bid_price", snap.iloc[0].get("last_price", leg.price)))
                    else:
                        close_leg.price = float(snap.iloc[0].get("ask_price", snap.iloc[0].get("last_price", leg.price)))

            if close_leg.price <= 0:
                if close_dir == "SELL":
                    close_leg.price = 0.01
                    logger.warning(f"  [PRICE] {leg.code} bid=0, using $0.01 minimum")
                else:
                    logger.error(f"  [CLOSE SKIP] {leg.code} ask=0, cannot buy back at 0")
                    failed = True
                    break

            if self.dry_run:
                close_leg.fill_price = close_leg.price
                logger.info(f"  [DRY] {close_dir} {leg.qty}x {leg.code} @ ${close_leg.price:.2f}")
                closed_legs.append(leg)
            else:
                ok = self._place_leg(close_leg)
                if ok:
                    closed_legs.append(leg)
                else:
                    failed = True
                    logger.error(f"  [CLOSE FAIL] {close_dir} {leg.code} 平仓失败")
                    break

        if failed and closed_legs:
            # Remove successfully closed legs so next retry only handles remaining ones
            for leg in closed_legs:
                if leg in trade.legs:
                    trade.legs.remove(leg)
            remaining_codes = [l.code for l in trade.legs]
            logger.warning(f"  [PARTIAL CLOSE] 已平 {len(closed_legs)} 腿, "
                           f"剩余 {len(trade.legs)} 腿: {remaining_codes}")
            self._notify(
                f"<b>部分平仓警告</b> {trade.strategy}\n"
                f"标的: <code>{trade.underlying}</code>\n"
                f"已平: {len(closed_legs)} 腿\n"
                f"剩余: {', '.join(remaining_codes)}\n"
                f"<b>下一周期将重试平仓剩余腿</b>"
            )
            return False

        if failed:
            return False

        pnl = self._compute_trade_pnl(trade)
        trade.status = "closed"
        trade.close_timestamp = __import__("datetime").datetime.now().isoformat()
        trade.realized_pnl = pnl
        trade.close_reason = reason

        premium = abs(trade.net_premium()) if any(l.price > 0 for l in trade.legs) else trade.max_loss
        self.risk.on_trade_close(trade.strategy, premium, pnl,
                                 underlying=trade.underlying)

        if trade in self._open_trades:
            self._open_trades.remove(trade)

        if self._store and trade.trade_id:
            try:
                self._store.close_option_trade(trade.trade_id, pnl, reason)
                logger.info(f"  [DB] trade_id={trade.trade_id} 已更新为 closed")
            except Exception as e:
                logger.error(f"  [DB] 更新失败: {e}")

        logger.info(f"  PnL: ${pnl:+.2f} | Reason: {reason}")
        self._notify_close(trade, pnl, reason)
        return True

    def _compute_trade_pnl(self, trade: OptionTrade) -> float:
        """Estimate PnL from fill prices vs current prices."""
        pnl = 0.0
        for leg in trade.legs:
            current = leg.fill_price
            if self._quote_ctx:
                from futu import RET_OK
                ret, snap = self._quote_ctx.get_market_snapshot([leg.code])
                if ret == RET_OK and snap is not None and len(snap) > 0:
                    current = float(snap.iloc[0].get("last_price", leg.fill_price))
            if leg.direction == "BUY":
                pnl += (current - leg.fill_price) * leg.qty * 100
            else:
                pnl += (leg.fill_price - current) * leg.qty * 100
        return round(pnl, 2)

    def monitor_open_trades(self, max_holding_days: int = 45,
                            close_before_expiry_days: int = 1):
        """Check stop-loss / take-profit / expiry on all open trades."""
        now = datetime.now()

        for trade in list(self._open_trades):
            if trade.status != "open":
                continue

            # ── Expiry guard: close before expiration ──
            earliest_expiry = self._get_earliest_expiry(trade)
            if earliest_expiry:
                days_to_expiry = (earliest_expiry - now.date()).days
                if days_to_expiry <= close_before_expiry_days:
                    self.close_trade(
                        trade,
                        f"Expiry guard: {days_to_expiry}d to expiry "
                        f"(threshold={close_before_expiry_days}d)")
                    continue

            # ── Max holding days ──
            if trade.open_timestamp and trade.open_timestamp != "restored":
                try:
                    open_dt = datetime.fromisoformat(trade.open_timestamp)
                    held_days = (now - open_dt).days
                    if held_days >= max_holding_days:
                        self.close_trade(
                            trade,
                            f"Max holding {held_days}d >= {max_holding_days}d")
                        continue
                except (ValueError, TypeError):
                    pass

            # ── Stop-loss / Take-profit ──
            pnl = self._compute_trade_pnl(trade)
            premium = abs(trade.net_premium()) if any(l.price > 0 for l in trade.legs) else trade.max_loss
            if premium <= 0:
                continue

            pnl_pct = pnl / premium if premium > 0 else 0

            if pnl <= -premium * trade.stop_loss_pct:
                self.close_trade(trade, f"Stop loss ({pnl_pct:.0%})")
            elif pnl >= premium * trade.take_profit_pct:
                self.close_trade(trade, f"Take profit ({pnl_pct:.0%})")

    @staticmethod
    def _get_earliest_expiry(trade: OptionTrade):
        """Return the earliest expiry date among all legs, or None."""
        from datetime import date as _date
        earliest = None
        for leg in trade.legs:
            if not leg.expiry:
                continue
            try:
                exp = datetime.strptime(leg.expiry, "%Y-%m-%d").date()
                if earliest is None or exp < earliest:
                    earliest = exp
            except ValueError:
                continue
        return earliest

    def _rollback_filled_legs(self, filled_legs: list[OptionLeg]):
        """Immediately close already-filled legs when a multi-leg trade partially fails."""
        logger.warning(f"  [ROLLBACK] Closing {len(filled_legs)} filled leg(s)...")
        for leg in filled_legs:
            close_dir = "SELL" if leg.direction == "BUY" else "BUY"
            if self._quote_ctx:
                from futu import RET_OK
                time.sleep(3.0)
                ret, snap = self._quote_ctx.get_market_snapshot([leg.code])
                if ret == RET_OK and snap is not None and len(snap) > 0:
                    if close_dir == "SELL":
                        price = float(snap.iloc[0].get("bid_price", leg.price))
                    else:
                        price = float(snap.iloc[0].get("ask_price", leg.price))
                else:
                    price = leg.price
            else:
                price = leg.price

            if price <= 0:
                price = leg.price * 0.95 if close_dir == "SELL" else leg.price * 1.05

            rollback_leg = OptionLeg(
                code=leg.code, underlying=leg.underlying,
                direction=close_dir, qty=leg.qty,
                option_type=leg.option_type, strike=leg.strike,
                expiry=leg.expiry, price=price,
            )
            ok = self._place_leg(rollback_leg)
            if ok:
                loss = abs(leg.price - price) * leg.qty * 100
                logger.warning(f"  [ROLLBACK OK] {close_dir} {leg.code} @ ${price:.2f} "
                               f"(est loss: ${loss:.2f})")
                self._notify(
                    f"<b>回滚平仓</b> {leg.direction} {leg.qty}张 "
                    f"<code>{leg.code}</code>\n"
                    f"平仓价: ${price:.2f} | 预估损失: ${loss:.2f}"
                )
            else:
                logger.error(f"  [ROLLBACK FAIL] Could not close {leg.code}! "
                             f"Manual intervention required.")
                self._notify(
                    f"<b>回滚失败！</b>\n"
                    f"无法平仓 <code>{leg.code}</code>\n"
                    f"请手动处理！"
                )

    def get_open_trades(self) -> list[OptionTrade]:
        return list(self._open_trades)

    # ── Telegram notifications ──

    def _notify(self, text: str):
        if self._notifier:
            try:
                self._notifier.send_sync(text)
            except Exception as e:
                logger.debug(f"Telegram notify failed: {e}")

    def _notify_open(self, trade: OptionTrade):
        mode = "模拟" if self.dry_run else "实盘"
        dir_cn = {"BUY": "买入", "SELL": "卖出"}
        legs_str = ""
        for leg in trade.legs:
            d = dir_cn.get(leg.direction, leg.direction)
            legs_str += f"\n  {d} {leg.qty}张 <code>{leg.code}</code> @ ${leg.price:.2f}"
        direction_str = ""
        if getattr(trade, "direction_details", ""):
            direction_str = f"\n\n<b>方向分析:</b>\n{trade.direction_details}"
        self._notify(
            f"<b>开仓 [{mode}]</b> {trade.strategy}\n"
            f"标的: <code>{trade.underlying}</code>"
            f"{legs_str}\n"
            f"最大亏损: ${trade.max_loss:.0f}"
            f"{direction_str}"
        )

    def _notify_close(self, trade: OptionTrade, pnl: float, reason: str):
        mode = " [模拟]" if self.dry_run else ""
        self._notify(
            f"<b>平仓{mode}</b> {trade.strategy}\n"
            f"标的: <code>{trade.underlying}</code>\n"
            f"盈亏: ${pnl:+.2f}\n"
            f"原因: {reason}"
        )
