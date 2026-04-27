from __future__ import annotations
from typing import Optional
from options.order import OptionLeg, OptionTrade
from options.chain import OptionChainFetcher
from utils.logger import setup_logger

logger = setup_logger("strategy.wheel")


class WheelStrategy:
    """Wheel strategy: Cash-Secured Put → assignment → Covered Call → called away → repeat.

    Phase 1 (CSP): Sell OTM Put on a stock priced ≤ $30.
                    If assigned, move to Phase 2.
    Phase 2 (CC):  Own 100 shares. Sell OTM Call against them.
                    If called away, return to Phase 1.

    This module handles Phase 1 (CSP) signal generation. Phase 2 requires
    owning shares — triggered manually or via position monitoring.
    """

    def __init__(self, config: dict, chain: OptionChainFetcher,
                 min_credit: float = 0.15):
        self.config = config
        self.chain = chain
        self.symbols = config.get("symbols", [])
        self.max_stock_price = config.get("max_stock_price", 30)
        self.csp_delta = config.get("csp_delta", 0.30)
        self.cc_delta = config.get("cc_delta", 0.30)
        self.min_ivr = config.get("min_ivr", 30)
        self.target_dte = config.get("target_dte", 30)
        self.profit_take_pct = config.get("profit_take_pct", 0.50)
        self.max_holding_days = config.get("max_holding_days", 45)
        self.min_credit = min_credit

    def evaluate_csp(self, symbol: str, ivr: float,
                     current_price: float) -> Optional[OptionTrade]:
        """Generate Cash-Secured Put trade for Wheel Phase 1."""
        if current_price > self.max_stock_price:
            logger.info(f"[WHEEL] {symbol} price ${current_price:.2f} "
                        f"> max ${self.max_stock_price}, skip")
            return None

        if ivr < self.min_ivr:
            logger.info(f"[WHEEL] {symbol} IVR={ivr:.1f} < min {self.min_ivr}, skip")
            return None

        cash_needed = current_price * 100
        logger.info(f"[WHEEL] {symbol} price=${current_price:.2f} IVR={ivr:.1f} "
                    f"cash_needed=${cash_needed:.0f}")

        # Find expiry ~30 days out
        expiries = self.chain.get_expiry_dates(symbol)
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=self.target_dte)
        target_expiry = None
        for exp in expiries:
            exp_date = exp.get("strike_time", "")[:10]
            if exp_date >= target_date.strftime("%Y-%m-%d"):
                target_expiry = exp_date
                break
        if target_expiry is None:
            logger.info(f"[WHEEL] {symbol} no suitable expiry found")
            return None

        # Find OTM Put at target delta
        short_put = self.chain.find_by_delta(symbol, target_expiry, "PUT", self.csp_delta)
        if short_put is None:
            logger.info(f"[WHEEL] {symbol} no put found for delta={self.csp_delta}")
            return None

        # Get quote for the put
        quotes = self.chain.get_option_quote([short_put["code"]])
        if not quotes or short_put["code"] not in quotes:
            return None

        bid = quotes[short_put["code"]]["bid_price"]
        if bid <= self.min_credit:
            logger.info(f"[WHEEL] {symbol} put bid ${bid:.2f} < min credit "
                        f"${self.min_credit:.2f}, skip")
            return None

        # CSP: sell 1 put, secured by cash
        sell_leg = OptionLeg(
            code=short_put["code"], underlying=symbol,
            direction="SELL", qty=1,
            option_type="PUT", strike=short_put["strike"],
            expiry=target_expiry, price=bid,
        )

        credit = bid * 100
        # Risk for portfolio tracking: assume stock drops ~20% from strike
        # (not full assignment cost — that would block all CSPs under $600 limit)
        downside_risk = short_put["strike"] * 100 * 0.20 - credit
        max_loss = max(downside_risk, credit * 2)  # at minimum 2x credit

        trade = OptionTrade(
            strategy="wheel_csp", underlying=symbol,
            legs=[sell_leg],
            max_loss=max_loss, target_pnl=credit * self.profit_take_pct,
            stop_loss_pct=2.00,
            take_profit_pct=self.profit_take_pct,
        )

        logger.info(f"[WHEEL CSP] {symbol}: Sell {short_put['strike']:.0f}P "
                    f"@ ${bid:.2f} | Credit ${credit:.0f} | "
                    f"Risk ${max_loss:.0f} (20% drop) | "
                    f"Assignment ${short_put['strike'] * 100:.0f} | "
                    f"IVR={ivr:.1f}")
        return trade

    def evaluate_cc(self, symbol: str, avg_cost: float,
                    shares: int = 100) -> Optional[OptionTrade]:
        """Generate Covered Call trade for Wheel Phase 2.

        Called when we own shares after CSP assignment.
        """
        if shares < 100:
            return None

        expiries = self.chain.get_expiry_dates(symbol)
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=self.target_dte)
        target_expiry = None
        for exp in expiries:
            exp_date = exp.get("strike_time", "")[:10]
            if exp_date >= target_date.strftime("%Y-%m-%d"):
                target_expiry = exp_date
                break
        if target_expiry is None:
            return None

        short_call = self.chain.find_by_delta(symbol, target_expiry, "CALL", self.cc_delta)
        if short_call is None:
            return None

        quotes = self.chain.get_option_quote([short_call["code"]])
        if not quotes or short_call["code"] not in quotes:
            return None

        bid = quotes[short_call["code"]]["bid_price"]
        if bid <= self.min_credit:
            return None

        sell_leg = OptionLeg(
            code=short_call["code"], underlying=symbol,
            direction="SELL", qty=1,
            option_type="CALL", strike=short_call["strike"],
            expiry=target_expiry, price=bid,
        )

        credit = bid * 100

        trade = OptionTrade(
            strategy="wheel_cc", underlying=symbol,
            legs=[sell_leg],
            max_loss=0,  # covered by shares
            target_pnl=credit * self.profit_take_pct,
            stop_loss_pct=2.00,
            take_profit_pct=self.profit_take_pct,
        )

        logger.info(f"[WHEEL CC] {symbol}: Sell {short_call['strike']:.0f}C "
                    f"@ ${bid:.2f} | Credit ${credit:.0f} | "
                    f"Avg cost ${avg_cost:.2f}")
        return trade
