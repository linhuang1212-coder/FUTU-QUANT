from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
from options.order import OptionLeg, OptionTrade
from options.chain import OptionChainFetcher
from utils.logger import setup_logger

logger = setup_logger("strategy.earnings")


class EarningsSpreadStrategy:
    """Bull Call Spread before earnings.
    
    Enter 5-7 days before earnings, exit 1 day before.
    Buy ATM Call + Sell OTM Call (5% higher strike).
    Conditions: IVR < 40%, technical uptrend.
    Max loss = net debit paid.
    """

    def __init__(self, config: dict, chain: OptionChainFetcher):
        self.config = config
        self.chain = chain
        self.symbols = config.get("scan_symbols", [])
        self.ivr_max = config.get("ivr_max", 40)
        self.days_before = config.get("days_before", 7)
        self.close_days_before = config.get("close_days_before", 1)
        self.spread_width_pct = config.get("spread_width_pct", 0.05)

    def evaluate(self, symbol: str, earnings_date: str,
                 current_ivr: float, is_uptrend: bool) -> Optional[OptionTrade]:
        """Generate earnings spread trade if conditions met."""
        earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        now = datetime.now()
        days_to_earnings = (earnings_dt - now).days

        if days_to_earnings < self.close_days_before or days_to_earnings > self.days_before:
            return None
        if current_ivr > self.ivr_max:
            logger.info(f"[EARNINGS] {symbol} IVR {current_ivr:.1f} > {self.ivr_max}, skip")
            return None
        if not is_uptrend:
            logger.info(f"[EARNINGS] {symbol} not in uptrend, skip")
            return None

        # Find expiry after earnings
        expiries = self.chain.get_expiry_dates(symbol)
        target_expiry = None
        for exp in expiries:
            exp_date = exp.get("strike_time", "")[:10]
            if exp_date >= earnings_date:
                target_expiry = exp_date
                break
        if target_expiry is None:
            return None

        # ATM Call (buy)
        atm = self.chain.find_atm_option(symbol, target_expiry, "CALL")
        if atm is None:
            return None

        # OTM Call (sell) - 5% higher strike
        otm_strike = atm["strike"] * (1 + self.spread_width_pct)
        chain_df = self.chain.get_chain(symbol, target_expiry, target_expiry, "CALL")
        if chain_df is None or len(chain_df) == 0:
            return None

        chain_df = chain_df.copy()
        chain_df["strike_diff"] = abs(chain_df["strike_price"] - otm_strike)
        otm_row = chain_df.sort_values("strike_diff").iloc[0]

        # Get prices
        codes = [atm["code"], otm_row["code"]]
        quotes = self.chain.get_option_quote(codes)
        if not quotes or len(quotes) < 2:
            return None

        atm_ask = quotes[atm["code"]]["ask_price"]
        otm_bid = quotes[otm_row["code"]]["bid_price"]
        if atm_ask <= 0 or otm_bid <= 0:
            return None

        net_debit = atm_ask - otm_bid
        if net_debit <= 0:
            return None

        buy_leg = OptionLeg(
            code=atm["code"], underlying=symbol,
            direction="BUY", qty=1,
            option_type="CALL", strike=atm["strike"],
            expiry=target_expiry, price=atm_ask,
        )
        sell_leg = OptionLeg(
            code=otm_row["code"], underlying=symbol,
            direction="SELL", qty=1,
            option_type="CALL", strike=float(otm_row["strike_price"]),
            expiry=target_expiry, price=otm_bid,
        )

        max_loss = net_debit * 100
        max_profit = (float(otm_row["strike_price"]) - atm["strike"] - net_debit) * 100

        trade = OptionTrade(
            strategy="earnings_spread", underlying=symbol,
            legs=[buy_leg, sell_leg],
            max_loss=max_loss, target_pnl=max_profit,
            stop_loss_pct=1.0, take_profit_pct=max_profit / max_loss if max_loss > 0 else 1.0,
        )

        logger.info(f"[EARNINGS] Bull Call Spread {symbol}: "
                    f"Buy {atm['strike']:.0f}C @ ${atm_ask:.2f}, "
                    f"Sell {otm_row['strike_price']:.0f}C @ ${otm_bid:.2f} "
                    f"| Net debit ${net_debit:.2f} | Max profit ${max_profit:.0f}")
        return trade
