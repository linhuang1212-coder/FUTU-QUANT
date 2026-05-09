from __future__ import annotations
from typing import Optional
from options.order import OptionLeg, OptionTrade
from options.chain import OptionChainFetcher
from utils.logger import setup_logger

logger = setup_logger("strategy.credit_spread")


class CreditSpreadStrategy:
    """Directional Credit Spread on high-IVR stocks.

    Bull Put Spread (direction=BULL): Sell OTM Put + Buy lower strike Put
    Bear Call Spread (direction=BEAR): Sell OTM Call + Buy higher strike Call

    Collect premium; keep it if stock stays on the right side of short strike.
    """

    # Dynamic spread width: scale with underlying price
    # Tight widths for small accounts — keep max_loss under $250
    SPREAD_WIDTH_TABLE = [
        (50,   1.0),   # stock <= $50:  $1 width  → max_loss ~$100
        (150,  2.5),   # stock <= $150: $2.5 width → max_loss ~$250
        (300,  2.5),   # stock <= $300: $2.5 width → max_loss ~$250
        (600,  5.0),   # stock <= $600: $5 width   → max_loss ~$500 (may exceed limit)
        (9999, 5.0),   # stock > $600:  $5 width
    ]

    MIN_RISK_REWARD_RATIO = 0.20  # net credit must be >= 20% of spread width

    def __init__(self, config: dict, chain: OptionChainFetcher,
                 min_credit: float = 0.30):
        self.config = config
        self.chain = chain
        self.target_delta = config.get("target_delta", 0.12)
        self.spread_width = config.get("spread_width", 5.0)
        self.profit_take_pct = config.get("profit_take_pct", 0.50)
        self.stop_loss_pct = config.get("stop_loss_pct", 2.00)
        self.max_holding_days = config.get("max_holding_days", 30)
        self.min_credit = min_credit

    def _dynamic_spread_width(self, stock_price: float) -> float:
        """Compute spread width based on underlying stock price."""
        for threshold, width in self.SPREAD_WIDTH_TABLE:
            if stock_price <= threshold:
                return width
        return self.spread_width

    def evaluate(self, symbol: str, ivr: float, min_ivr: float = 60.0,
                 target_dte: int = 21,
                 direction: str = "BULL") -> Optional[OptionTrade]:
        """Generate credit spread based on direction signal.

        direction="BULL" -> Bull Put Spread (sell put spread, bullish bet)
        direction="BEAR" -> Bear Call Spread (sell call spread, bearish bet)
        """
        if ivr < min_ivr:
            return None

        expiries = self.chain.get_expiry_dates(symbol)
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=target_dte)
        target_expiry = None
        for exp in expiries:
            exp_date = exp.get("strike_time", "")[:10]
            if exp_date >= target_date.strftime("%Y-%m-%d"):
                target_expiry = exp_date
                break
        if target_expiry is None:
            return None

        if direction == "BEAR":
            return self._evaluate_bear_call(symbol, ivr, target_expiry)
        else:
            return self._evaluate_bull_put(symbol, ivr, target_expiry)

    def _evaluate_bull_put(self, symbol: str, ivr: float,
                           target_expiry: str) -> Optional[OptionTrade]:
        """Bull Put Spread: sell higher put, buy lower put."""
        short_put = self.chain.find_by_delta(symbol, target_expiry, "PUT", self.target_delta)
        if short_put is None:
            logger.info(f"[CREDIT] {symbol} Bull Put: delta={self.target_delta} 的Put未找到")
            return None

        stock_price = short_put.get("stock_price", short_put["strike"] * 1.15)
        width = self._dynamic_spread_width(stock_price)

        logger.info(f"[CREDIT] {symbol} Bull Put: short strike={short_put['strike']:.0f} "
                    f"delta={short_put.get('delta', '?')} expiry={target_expiry} "
                    f"width=${width:.1f} (price=${stock_price:.0f})")

        long_strike = short_put["strike"] - width
        chain_df = self.chain.get_chain(symbol, target_expiry, target_expiry, "PUT")
        if chain_df is None or len(chain_df) == 0:
            return None

        chain_df = chain_df.copy()
        chain_df["strike_diff"] = abs(chain_df["strike_price"] - long_strike)
        long_row = chain_df.sort_values("strike_diff").iloc[0]

        codes = [short_put["code"], long_row["code"]]
        quotes = self.chain.get_option_quote(codes)
        if not quotes or len(quotes) < 2:
            return None

        short_bid = quotes[short_put["code"]]["bid_price"]
        long_ask = quotes[long_row["code"]]["ask_price"]

        logger.info(f"[CREDIT] {symbol} Bull Put 报价: "
                    f"卖 {short_put['strike']:.0f}P bid=${short_bid:.2f}, "
                    f"买 {long_row['strike_price']:.0f}P ask=${long_ask:.2f}")

        if short_bid <= 0 or long_ask <= 0:
            logger.info(f"[CREDIT] {symbol} Bull Put bid/ask 无效 "
                        f"(bid=${short_bid:.2f}, ask=${long_ask:.2f}), 跳过")
            return None

        net_credit = short_bid - long_ask
        actual_width = short_put["strike"] - float(long_row["strike_price"])

        # Quality gate 1: minimum absolute credit
        if net_credit <= self.min_credit:
            logger.info(f"[CREDIT] {symbol} Bull Put credit ${net_credit:.2f} "
                        f"< 最低 ${self.min_credit:.2f}, 跳过")
            return None

        # Quality gate 2: risk/reward ratio — credit must be >= 20% of width
        rr_ratio = net_credit / actual_width if actual_width > 0 else 0
        if rr_ratio < self.MIN_RISK_REWARD_RATIO:
            logger.info(f"[CREDIT] {symbol} Bull Put R/R {rr_ratio:.1%} "
                        f"< 最低 {self.MIN_RISK_REWARD_RATIO:.0%}, 跳过")
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

        max_loss = (actual_width - net_credit) * 100
        max_profit = net_credit * 100

        trade = OptionTrade(
            strategy="credit_spread", underlying=symbol,
            legs=[sell_leg, buy_leg],
            max_loss=max_loss, target_pnl=max_profit * self.profit_take_pct,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.profit_take_pct,
        )

        logger.info(f"[CREDIT] Bull Put Spread {symbol}: "
                    f"卖 {short_put['strike']:.0f}P @ ${short_bid:.2f}, "
                    f"买 {long_row['strike_price']:.0f}P @ ${long_ask:.2f} "
                    f"| Credit ${net_credit:.2f} ({rr_ratio:.0%} R/R) "
                    f"| 最大亏损 ${max_loss:.0f} | IVR={ivr:.1f}")
        return trade

    def _evaluate_bear_call(self, symbol: str, ivr: float,
                            target_expiry: str) -> Optional[OptionTrade]:
        """Bear Call Spread: sell lower call, buy higher call."""
        short_call = self.chain.find_by_delta(symbol, target_expiry, "CALL", self.target_delta)
        if short_call is None:
            logger.info(f"[CREDIT] {symbol} Bear Call: delta={self.target_delta} 的Call未找到")
            return None

        stock_price = short_call.get("stock_price", short_call["strike"] * 0.85)
        width = self._dynamic_spread_width(stock_price)

        logger.info(f"[CREDIT] {symbol} Bear Call: short strike={short_call['strike']:.0f} "
                    f"delta={short_call.get('delta', '?')} expiry={target_expiry} "
                    f"width=${width:.1f} (price=${stock_price:.0f})")

        long_strike = short_call["strike"] + width
        chain_df = self.chain.get_chain(symbol, target_expiry, target_expiry, "CALL")
        if chain_df is None or len(chain_df) == 0:
            return None

        chain_df = chain_df.copy()
        chain_df["strike_diff"] = abs(chain_df["strike_price"] - long_strike)
        long_row = chain_df.sort_values("strike_diff").iloc[0]

        codes = [short_call["code"], long_row["code"]]
        quotes = self.chain.get_option_quote(codes)
        if not quotes or len(quotes) < 2:
            return None

        short_bid = quotes[short_call["code"]]["bid_price"]
        long_ask = quotes[long_row["code"]]["ask_price"]

        logger.info(f"[CREDIT] {symbol} Bear Call 报价: "
                    f"卖 {short_call['strike']:.0f}C bid=${short_bid:.2f}, "
                    f"买 {long_row['strike_price']:.0f}C ask=${long_ask:.2f}")

        if short_bid <= 0 or long_ask <= 0:
            logger.info(f"[CREDIT] {symbol} Bear Call bid/ask 无效 "
                        f"(bid=${short_bid:.2f}, ask=${long_ask:.2f}), 跳过")
            return None

        net_credit = short_bid - long_ask
        actual_width = float(long_row["strike_price"]) - short_call["strike"]

        if net_credit <= self.min_credit:
            logger.info(f"[CREDIT] {symbol} Bear Call credit ${net_credit:.2f} "
                        f"< 最低 ${self.min_credit:.2f}, 跳过")
            return None

        rr_ratio = net_credit / actual_width if actual_width > 0 else 0
        if rr_ratio < self.MIN_RISK_REWARD_RATIO:
            logger.info(f"[CREDIT] {symbol} Bear Call R/R {rr_ratio:.1%} "
                        f"< 最低 {self.MIN_RISK_REWARD_RATIO:.0%}, 跳过")
            return None

        sell_leg = OptionLeg(
            code=short_call["code"], underlying=symbol,
            direction="SELL", qty=1,
            option_type="CALL", strike=short_call["strike"],
            expiry=target_expiry, price=short_bid,
        )
        buy_leg = OptionLeg(
            code=long_row["code"], underlying=symbol,
            direction="BUY", qty=1,
            option_type="CALL", strike=float(long_row["strike_price"]),
            expiry=target_expiry, price=long_ask,
        )

        max_loss = (actual_width - net_credit) * 100
        max_profit = net_credit * 100

        trade = OptionTrade(
            strategy="credit_spread", underlying=symbol,
            legs=[sell_leg, buy_leg],
            max_loss=max_loss, target_pnl=max_profit * self.profit_take_pct,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.profit_take_pct,
        )

        logger.info(f"[CREDIT] Bear Call Spread {symbol}: "
                    f"卖 {short_call['strike']:.0f}C @ ${short_bid:.2f}, "
                    f"买 {long_row['strike_price']:.0f}C @ ${long_ask:.2f} "
                    f"| Credit ${net_credit:.2f} ({rr_ratio:.0%} R/R) "
                    f"| 最大亏损 ${max_loss:.0f} | IVR={ivr:.1f}")
        return trade
