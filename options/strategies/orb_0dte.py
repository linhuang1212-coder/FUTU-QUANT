from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Optional
from options.order import OptionLeg, OptionTrade
from options.chain import OptionChainFetcher
from utils.logger import setup_logger

logger = setup_logger("strategy.orb_0dte")


class ORBStrategy:
    """0DTE Opening Range Breakout.
    
    9:30-10:00: Record 30-min high (H) and low (L)
    After 10:00: Break above H -> buy ATM Call; break below L -> buy ATM Put
    Stop: option value drops 50%
    Target: option value doubles (100% gain)
    Max premium per trade: $200-300
    """

    def __init__(self, config: dict, chain: OptionChainFetcher,
                 max_loss_per_trade: float = 200):
        self.config = config
        self.chain = chain
        self.symbols = config.get("symbols", ["US.SPY", "US.QQQ"])
        self.orb_minutes = config.get("orb_minutes", 30)
        self.max_premium = config.get("max_premium", 300)
        self.max_loss_per_trade = max_loss_per_trade
        self.stop_loss_pct = config.get("stop_loss_pct", 0.50)
        self.target_pct = config.get("target_pct", 1.00)
        
        # ORB state per symbol
        self._orb_high: dict[str, float] = {}
        self._orb_low: dict[str, float] = {}
        self._orb_ready: dict[str, bool] = {}
        self._triggered: dict[str, bool] = {}

    def reset_daily(self):
        """Reset at start of each trading day."""
        self._orb_high.clear()
        self._orb_low.clear()
        self._orb_ready = {s: False for s in self.symbols}
        self._triggered = {s: False for s in self.symbols}

    def update_orb(self, symbol: str, high: float, low: float, bar_time: datetime):
        """Update ORB range during 9:30-10:00 window."""
        from zoneinfo import ZoneInfo
        et = bar_time.astimezone(ZoneInfo("US/Eastern")) if bar_time.tzinfo else bar_time
        market_open = et.replace(hour=9, minute=30, second=0)
        orb_end = market_open + timedelta(minutes=self.orb_minutes)

        if et < market_open or et >= orb_end:
            if et >= orb_end and symbol in self._orb_high:
                self._orb_ready[symbol] = True
            return

        if symbol not in self._orb_high:
            self._orb_high[symbol] = high
            self._orb_low[symbol] = low
        else:
            self._orb_high[symbol] = max(self._orb_high[symbol], high)
            self._orb_low[symbol] = min(self._orb_low[symbol], low)

    def evaluate(self, symbol: str, current_price: float) -> Optional[OptionTrade]:
        """Check for ORB breakout after 10:00 ET."""
        if not self._orb_ready.get(symbol, False):
            return None
        if self._triggered.get(symbol, False):
            return None

        orb_h = self._orb_high.get(symbol)
        orb_l = self._orb_low.get(symbol)
        if orb_h is None or orb_l is None:
            return None

        direction = None
        if current_price > orb_h:
            direction = "CALL"
            logger.info(f"[ORB] {symbol} broke above H={orb_h:.2f} (price={current_price:.2f})")
        elif current_price < orb_l:
            direction = "PUT"
            logger.info(f"[ORB] {symbol} broke below L={orb_l:.2f} (price={current_price:.2f})")

        if direction is None:
            return None

        self._triggered[symbol] = True
        today = datetime.now().strftime("%Y-%m-%d")
        atm = self.chain.find_atm_option(symbol, today, direction)
        if atm is None:
            logger.warning(f"[ORB] No 0DTE {direction} found for {symbol}")
            return None

        # Get option price
        quote = self.chain.get_option_quote([atm["code"]])
        if not quote or atm["code"] not in quote:
            return None
        ask = quote[atm["code"]]["ask_price"]
        if ask <= 0:
            ask = quote[atm["code"]]["last_price"]

        # Dynamic sizing: max_loss / (option_price * 100 * stop_loss_pct)
        # e.g. ask=$3.0, stop=50% -> risk per contract = $150
        # Capped by both max_loss_per_trade and max_premium
        risk_per_contract = ask * 100 * self.stop_loss_pct
        budget = min(self.max_loss_per_trade, self.max_premium)
        contracts = max(1, int(budget / risk_per_contract)) if risk_per_contract > 0 else 1
        premium = ask * contracts * 100
        # Final check: total premium must not exceed risk limit
        if premium > self.max_loss_per_trade:
            contracts = 1
            premium = ask * 100

        leg = OptionLeg(
            code=atm["code"], underlying=symbol,
            direction="BUY", qty=contracts,
            option_type=direction, strike=atm["strike"],
            expiry=today, price=ask,
        )

        trade = OptionTrade(
            strategy="orb_0dte", underlying=symbol,
            legs=[leg], max_loss=premium,
            target_pnl=premium * self.target_pct,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.target_pct,
        )

        logger.info(f"[ORB SIGNAL] {direction} {symbol} | {atm['code']} "
                    f"strike={atm['strike']} @ ${ask:.2f} x{contracts}")
        return trade
