from __future__ import annotations
from typing import Optional
from options.order import OptionLeg, OptionTrade
from options.chain import OptionChainFetcher
from utils.logger import setup_logger

logger = setup_logger("strategy.pmcc")


class PMCCStrategy:
    """Poor Man's Covered Call.

    Buy a deep-ITM LEAPS Call (delta ~0.80, 1 year out) as a stock substitute,
    then sell near-month OTM Calls to collect premium.

    Phase 1 (LEAPS): Buy deep ITM Call ~1 year out.
    Phase 2 (Short Call): Sell OTM Call ~30 days out against the LEAPS.
    """

    def __init__(self, config: dict, chain: OptionChainFetcher):
        self.config = config
        self.chain = chain
        self.symbols = config.get("symbols", [])
        self.leaps_delta = config.get("leaps_delta", 0.80)
        self.leaps_dte = config.get("leaps_dte", 365)
        self.short_call_delta = config.get("short_call_delta", 0.30)
        self.short_call_dte = config.get("short_call_dte", 30)
        self.max_leaps_cost = config.get("max_leaps_cost", 2500)
        self.profit_take_pct = config.get("profit_take_pct", 0.50)

    def evaluate_leaps(self, symbol: str) -> Optional[OptionTrade]:
        """Phase 1: Buy deep ITM LEAPS Call."""
        expiries = self.chain.get_expiry_dates(symbol)
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=self.leaps_dte)

        leaps_expiry = None
        for exp in expiries:
            exp_date = exp.get("strike_time", "")[:10]
            if exp_date >= target_date.strftime("%Y-%m-%d"):
                leaps_expiry = exp_date
                break
        if leaps_expiry is None:
            logger.info(f"[PMCC] {symbol} no LEAPS expiry found ≥ {self.leaps_dte}d out")
            return None

        # Find deep ITM Call (high delta)
        leaps = self.chain.find_by_delta(symbol, leaps_expiry, "CALL", self.leaps_delta)
        if leaps is None:
            logger.info(f"[PMCC] {symbol} no LEAPS call found for delta={self.leaps_delta}")
            return None

        quotes = self.chain.get_option_quote([leaps["code"]])
        if not quotes or leaps["code"] not in quotes:
            return None

        ask = quotes[leaps["code"]]["ask_price"]
        cost = ask * 100
        if cost > self.max_leaps_cost:
            logger.info(f"[PMCC] {symbol} LEAPS cost ${cost:.0f} > max ${self.max_leaps_cost}")
            return None

        buy_leg = OptionLeg(
            code=leaps["code"], underlying=symbol,
            direction="BUY", qty=1,
            option_type="CALL", strike=leaps["strike"],
            expiry=leaps_expiry, price=ask,
        )

        trade = OptionTrade(
            strategy="pmcc_leaps", underlying=symbol,
            legs=[buy_leg],
            max_loss=cost,
            target_pnl=0,  # LEAPS is a long-term hold
            stop_loss_pct=0.40,
            take_profit_pct=1.00,
        )

        logger.info(f"[PMCC LEAPS] {symbol}: Buy {leaps['strike']:.0f}C "
                    f"exp={leaps_expiry} @ ${ask:.2f} (cost=${cost:.0f}, "
                    f"delta={leaps.get('delta', '?')})")
        return trade

    def evaluate_short_call(self, symbol: str,
                            leaps_strike: float) -> Optional[OptionTrade]:
        """Phase 2: Sell OTM Call against existing LEAPS."""
        expiries = self.chain.get_expiry_dates(symbol)
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=self.short_call_dte)

        short_expiry = None
        for exp in expiries:
            exp_date = exp.get("strike_time", "")[:10]
            if exp_date >= target_date.strftime("%Y-%m-%d"):
                short_expiry = exp_date
                break
        if short_expiry is None:
            return None

        short_call = self.chain.find_by_delta(
            symbol, short_expiry, "CALL", self.short_call_delta)
        if short_call is None:
            return None

        # Short call strike must be above LEAPS strike for diagonal spread
        if short_call["strike"] <= leaps_strike:
            logger.info(f"[PMCC] {symbol} short call strike {short_call['strike']:.0f} "
                        f"≤ LEAPS strike {leaps_strike:.0f}, skip")
            return None

        quotes = self.chain.get_option_quote([short_call["code"]])
        if not quotes or short_call["code"] not in quotes:
            return None

        bid = quotes[short_call["code"]]["bid_price"]
        if bid <= 0.10:
            return None

        sell_leg = OptionLeg(
            code=short_call["code"], underlying=symbol,
            direction="SELL", qty=1,
            option_type="CALL", strike=short_call["strike"],
            expiry=short_expiry, price=bid,
        )

        credit = bid * 100
        trade = OptionTrade(
            strategy="pmcc_short", underlying=symbol,
            legs=[sell_leg],
            max_loss=0,  # covered by LEAPS
            target_pnl=credit * self.profit_take_pct,
            stop_loss_pct=2.00,
            take_profit_pct=self.profit_take_pct,
        )

        logger.info(f"[PMCC SHORT] {symbol}: Sell {short_call['strike']:.0f}C "
                    f"exp={short_expiry} @ ${bid:.2f} | Credit ${credit:.0f}")
        return trade
