from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
from options.order import OptionLeg, OptionTrade
from options.chain import OptionChainFetcher
from utils.logger import setup_logger

logger = setup_logger("strategy.straddle")


class StraddleSqueezeStrategy:
    """Buy ATM Straddle when Bollinger Band Width is at historical lows.
    
    Trigger: BBW at 6-month low (bottom 5th percentile).
    Entry: Buy ATM Call + ATM Put (same strike, same expiry).
    Exit: After directional breakout, close losing side.
    Risk: Theta decay if no breakout occurs.
    """

    def __init__(self, config: dict, chain: OptionChainFetcher):
        self.config = config
        self.chain = chain
        self.symbols = config.get("symbols", ["US.GLD", "US.TLT"])
        self.max_holding_days = config.get("max_holding_days", 14)

    def evaluate(self, symbol: str, squeeze_data: dict) -> Optional[OptionTrade]:
        """Generate straddle trade if BB squeeze is detected."""
        if not squeeze_data:
            return None

        logger.info(f"[STRADDLE] BB squeeze on {symbol}: "
                    f"width={squeeze_data['bb_width']:.4f}, "
                    f"rank={squeeze_data['percentile_rank']:.0f}%")

        # Find expiry 2-3 weeks out (give time for breakout)
        expiries = self.chain.get_expiry_dates(symbol)
        target_date = datetime.now() + timedelta(days=self.max_holding_days + 7)
        target_expiry = None
        for exp in expiries:
            exp_date = exp.get("strike_time", "")[:10]
            if exp_date >= target_date.strftime("%Y-%m-%d"):
                target_expiry = exp_date
                break
        if target_expiry is None:
            return None

        # ATM Call
        atm_call = self.chain.find_atm_option(symbol, target_expiry, "CALL")
        if atm_call is None:
            return None

        # ATM Put at same strike
        chain_df = self.chain.get_chain(symbol, target_expiry, target_expiry, "PUT")
        if chain_df is None or len(chain_df) == 0:
            return None
        chain_df = chain_df.copy()
        chain_df["strike_diff"] = abs(chain_df["strike_price"] - atm_call["strike"])
        put_row = chain_df.sort_values("strike_diff").iloc[0]

        # Get prices
        codes = [atm_call["code"], put_row["code"]]
        quotes = self.chain.get_option_quote(codes)
        if not quotes or len(quotes) < 2:
            return None

        call_ask = quotes[atm_call["code"]]["ask_price"]
        put_ask = quotes[put_row["code"]]["ask_price"]
        if call_ask <= 0 or put_ask <= 0:
            return None

        total_premium = (call_ask + put_ask) * 100

        call_leg = OptionLeg(
            code=atm_call["code"], underlying=symbol,
            direction="BUY", qty=1,
            option_type="CALL", strike=atm_call["strike"],
            expiry=target_expiry, price=call_ask,
        )
        put_leg = OptionLeg(
            code=put_row["code"], underlying=symbol,
            direction="BUY", qty=1,
            option_type="PUT", strike=float(put_row["strike_price"]),
            expiry=target_expiry, price=put_ask,
        )

        trade = OptionTrade(
            strategy="straddle", underlying=symbol,
            legs=[call_leg, put_leg],
            max_loss=total_premium,
            target_pnl=total_premium * 0.5,
            stop_loss_pct=0.50,
            take_profit_pct=0.50,
        )

        logger.info(f"[STRADDLE] ATM Straddle {symbol} @ {atm_call['strike']:.0f}: "
                    f"Call ${call_ask:.2f} + Put ${put_ask:.2f} = ${total_premium:.0f}")
        return trade
