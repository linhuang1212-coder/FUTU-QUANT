from __future__ import annotations
import time
from typing import Optional
from options.order import OptionLeg, OptionTrade
from options.risk import OptionsRiskManager, EarningsGuard
from utils.logger import setup_logger

logger = setup_logger("options.trader")


class OptionsTrader:
    """Execute option trades via Futu OpenAPI."""

    def __init__(self, trade_ctx, quote_ctx, risk_mgr: OptionsRiskManager,
                 trade_env: str = "SIMULATE", dry_run: bool = False,
                 fmp_api_key: str = ""):
        self._trade_ctx = trade_ctx
        self._quote_ctx = quote_ctx
        self.risk = risk_mgr
        self.earnings_guard = EarningsGuard(quote_ctx, fmp_api_key=fmp_api_key)
        self.trade_env = trade_env
        self.dry_run = dry_run
        self._open_trades: list[OptionTrade] = []

    def open_trade(self, trade: OptionTrade) -> bool:
        """Open a multi-leg option trade."""
        # Earnings check for individual stocks
        if trade.legs:
            expiry = trade.legs[0].expiry
            safe, reason = self.earnings_guard.check(trade.underlying, expiry)
            if not safe:
                logger.warning(f"[EARNINGS BLOCK] {trade.strategy}: {reason}")
                return False

        premium = abs(trade.net_premium()) if any(l.price > 0 for l in trade.legs) else trade.max_loss
        ok, reason = self.risk.can_open_trade(
            trade.strategy, premium,
            underlying=trade.underlying, max_loss=trade.max_loss,
        )
        if not ok:
            logger.warning(f"[RISK BLOCK] {trade.strategy}: {reason}")
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
            return True

        all_ok = True
        for leg in trade.legs:
            success = self._place_leg(leg)
            if not success:
                all_ok = False
                logger.error(f"  [FAIL] {leg.direction} {leg.code}")
                break
            time.sleep(0.5)

        if all_ok:
            trade.status = "open"
            trade.open_timestamp = __import__("datetime").datetime.now().isoformat()
            self.risk.on_trade_open(trade.strategy, premium,
                                    underlying=trade.underlying,
                                    max_loss=trade.max_loss)
            self._open_trades.append(trade)
        return all_ok

    def _place_leg(self, leg: OptionLeg) -> bool:
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
        """Close all legs of an open trade."""
        if trade.status != "open":
            return False

        logger.info(f"[CLOSE] {trade.strategy} on {trade.underlying}: {reason}")

        for leg in trade.legs:
            close_dir = "SELL" if leg.direction == "BUY" else "BUY"
            close_leg = OptionLeg(
                code=leg.code, underlying=leg.underlying,
                direction=close_dir, qty=leg.qty,
                option_type=leg.option_type, strike=leg.strike,
                expiry=leg.expiry, price=0,
            )
            if self._quote_ctx:
                from futu import RET_OK
                time.sleep(0.3)
                ret, snap = self._quote_ctx.get_market_snapshot([leg.code])
                if ret == RET_OK and snap is not None and len(snap) > 0:
                    close_leg.price = float(snap.iloc[0].get("last_price", leg.price))

            if self.dry_run:
                close_leg.fill_price = close_leg.price
                logger.info(f"  [DRY] {close_dir} {leg.qty}x {leg.code} @ ${close_leg.price:.2f}")
            else:
                self._place_leg(close_leg)

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

        logger.info(f"  PnL: ${pnl:+.2f} | Reason: {reason}")
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

    def monitor_open_trades(self):
        """Check stop-loss / take-profit on all open trades."""
        for trade in list(self._open_trades):
            if trade.status != "open":
                continue
            pnl = self._compute_trade_pnl(trade)
            premium = abs(trade.net_premium()) if any(l.price > 0 for l in trade.legs) else trade.max_loss
            if premium <= 0:
                continue

            pnl_pct = pnl / premium if premium > 0 else 0

            if pnl <= -premium * trade.stop_loss_pct:
                self.close_trade(trade, f"Stop loss ({pnl_pct:.0%})")
            elif pnl >= premium * trade.take_profit_pct:
                self.close_trade(trade, f"Take profit ({pnl_pct:.0%})")

    def get_open_trades(self) -> list[OptionTrade]:
        return list(self._open_trades)
