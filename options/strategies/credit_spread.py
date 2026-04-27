from __future__ import annotations
from typing import Optional
from options.order import OptionLeg, OptionTrade
from options.chain import OptionChainFetcher
from utils.logger import setup_logger

logger = setup_logger("strategy.credit_spread")


class CreditSpreadStrategy:
    """Bull Put Credit Spread on high-IVR stocks.
    
    Sell OTM Put (Delta ~0.3) + Buy lower strike Put for protection.
    Collect premium; keep it if stock stays above short strike.
    Take profit at 50% of premium received.
    Stop loss at 200% of premium received.
    Hold 7-21 days.
    """

    def __init__(self, config: dict, chain: OptionChainFetcher,
                 min_credit: float = 0.15):
        self.config = config
        self.chain = chain
        self.target_delta = config.get("target_delta", 0.30)
        self.spread_width = config.get("spread_width", 5.0)
        self.profit_take_pct = config.get("profit_take_pct", 0.50)
        self.stop_loss_pct = config.get("stop_loss_pct", 2.00)
        self.max_holding_days = config.get("max_holding_days", 21)
        self.min_credit = min_credit

    def evaluate(self, symbol: str, ivr: float, min_ivr: float = 60.0) -> Optional[OptionTrade]:
        """Generate credit spread if IVR is high enough."""
        if ivr < min_ivr:
            return None

        # Find expiry 2-4 weeks out
        expiries = self.chain.get_expiry_dates(symbol)
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=21)
        target_expiry = None
        for exp in expiries:
            exp_date = exp.get("strike_time", "")[:10]
            if exp_date >= target_date.strftime("%Y-%m-%d"):
                target_expiry = exp_date
                break
        if target_expiry is None:
            return None

        # Short Put: Delta ~0.3
        short_put = self.chain.find_by_delta(symbol, target_expiry, "PUT", self.target_delta)
        if short_put is None:
            logger.info(f"[CREDIT] {symbol} no put found for delta={self.target_delta}")
            return None

        logger.info(f"[CREDIT] {symbol} short put: strike={short_put['strike']:.0f} "
                    f"delta={short_put.get('delta', '?')} expiry={target_expiry}")

        # Long Put: lower strike for protection
        long_strike = short_put["strike"] - self.spread_width
        chain_df = self.chain.get_chain(symbol, target_expiry, target_expiry, "PUT")
        if chain_df is None or len(chain_df) == 0:
            return None

        chain_df = chain_df.copy()
        chain_df["strike_diff"] = abs(chain_df["strike_price"] - long_strike)
        long_row = chain_df.sort_values("strike_diff").iloc[0]

        # Get prices
        codes = [short_put["code"], long_row["code"]]
        quotes = self.chain.get_option_quote(codes)
        if not quotes or len(quotes) < 2:
            return None

        short_bid = quotes[short_put["code"]]["bid_price"]
        short_ask = quotes[short_put["code"]]["ask_price"]
        long_bid = quotes[long_row["code"]]["bid_price"]
        long_ask = quotes[long_row["code"]]["ask_price"]

        logger.info(f"[CREDIT] {symbol} prices: "
                    f"short {short_put['strike']:.0f}P bid=${short_bid:.2f}/ask=${short_ask:.2f}, "
                    f"long {long_row['strike_price']:.0f}P bid=${long_bid:.2f}/ask=${long_ask:.2f}")

        if short_bid <= 0:
            logger.info(f"[CREDIT] {symbol} short put has no bid, skip")
            return None

        net_credit = short_bid - long_ask
        if net_credit <= self.min_credit:
            logger.info(f"[CREDIT] {symbol} net credit ${net_credit:.2f} "
                        f"< min ${self.min_credit:.2f}, skip")
            return None

        sell_leg = OptionLeg(
            code=short_put["code"], underlying=symbol,
            direction="SELL", qty=1,
            option_type="PUT", strike=short_put["strike"],
            expiry=target_expiry, price=short_bid,
        )
        buy_leg = OptionLeg(
            code=long_row["code"], underlying=symbol,
            direction="BUY", qty=1,
            option_type="PUT", strike=float(long_row["strike_price"]),
            expiry=target_expiry, price=long_ask,
        )

        max_loss = (short_put["strike"] - float(long_row["strike_price"]) - net_credit) * 100
        max_profit = net_credit * 100

        trade = OptionTrade(
            strategy="credit_spread", underlying=symbol,
            legs=[sell_leg, buy_leg],
            max_loss=max_loss, target_pnl=max_profit * self.profit_take_pct,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.profit_take_pct,
        )

        logger.info(f"[CREDIT] Bull Put Spread {symbol}: "
                    f"Sell {short_put['strike']:.0f}P @ ${short_bid:.2f}, "
                    f"Buy {long_row['strike_price']:.0f}P @ ${long_ask:.2f} "
                    f"| Credit ${net_credit:.2f} | Max loss ${max_loss:.0f} "
                    f"| IVR={ivr:.1f}")
        return trade
