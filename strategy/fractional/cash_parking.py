"""
Cash Parking Strategy — 闲置资金自动停泊 SGOV.

Monitors idle cash in the account and automatically:
  - Buys SGOV when idle cash > threshold
  - Sells SGOV when options need capital (cash drops below safety buffer)

Since Moomoo OpenAPI does NOT support fractional shares,
all orders use whole-share quantities (SGOV ~$100/share).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from futu import (
    OpenQuoteContext, OpenSecTradeContext, RET_OK,
    TrdEnv, TrdSide, OrderType,
)

logger = logging.getLogger(__name__)


@dataclass
class ParkingConfig:
    symbol: str = "US.SGOV"
    fallback_symbol: str = "US.BIL"
    min_idle_to_buy: float = 150
    safety_buffer: float = 300
    trade_env: str = "REAL"

    @classmethod
    def from_yaml(cls, cfg: dict) -> ParkingConfig:
        cp = cfg.get("cash_parking", {})
        return cls(
            symbol=cp.get("symbol", "US.SGOV"),
            fallback_symbol=cp.get("fallback_symbol", "US.BIL"),
            min_idle_to_buy=cp.get("min_idle_to_buy", 150),
            safety_buffer=cp.get("safety_buffer", 300),
        )


class CashParking:
    """Manages idle cash by parking in short-term treasury ETF."""

    def __init__(self, config: ParkingConfig,
                 quote_ctx: OpenQuoteContext,
                 trade_ctx: OpenSecTradeContext,
                 notifier=None):
        self.cfg = config
        self.quote_ctx = quote_ctx
        self.trade_ctx = trade_ctx
        self.notifier = notifier
        self.trd_env = TrdEnv.REAL if config.trade_env == "REAL" else TrdEnv.SIMULATE

    def get_account_cash(self) -> float:
        """Return available buying power (not frozen by margin)."""
        ret, data = self.trade_ctx.accinfo_query(trd_env=self.trd_env)
        if ret != RET_OK or data.empty:
            logger.warning("无法查询账户资金: %s", data)
            return 0.0
        # Use available_funds (buying power) instead of raw cash
        # to account for margin frozen by options positions
        available = data.get("available_funds")
        if available is not None and not available.empty:
            val = available.iloc[0]
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return float(data["cash"].iloc[0])

    def get_position(self, symbol: str) -> int:
        """Get current whole-share holding of symbol."""
        ret, data = self.trade_ctx.position_list_query(trd_env=self.trd_env)
        if ret != RET_OK or data.empty:
            return 0
        row = data[data["code"] == symbol]
        if row.empty:
            return 0
        return int(row["qty"].iloc[0])

    def get_price(self, symbol: str) -> float:
        ret, snap = self.quote_ctx.get_market_snapshot([symbol])
        if ret != RET_OK or snap.empty:
            return 0.0
        return float(snap["last_price"].iloc[0])

    def check_and_park(self, options_capital_needed: float = 0) -> Optional[dict]:
        """Main loop entry: check idle cash and park/unpark as needed.

        Args:
            options_capital_needed: extra cash the options engine needs soon
                                   (e.g., pending spread margin). Set > 0 to
                                   trigger a sell.

        Returns dict with action taken, or None.
        """
        cash = self.get_account_cash()
        held = self.get_position(self.cfg.symbol)
        price = self.get_price(self.cfg.symbol)

        if price <= 0:
            logger.warning("无法获取 %s 报价", self.cfg.symbol)
            return None

        held_value = held * price
        idle = cash - self.cfg.safety_buffer - options_capital_needed

        logger.info("Cash Parking: 现金=$%.2f | %s持仓=%d股($%.0f) | 闲置=$%.0f",
                     cash, self.cfg.symbol, held, held_value, idle)

        # Sell SGOV if cash is below safety buffer (options need capital)
        if cash < self.cfg.safety_buffer and held > 0:
            sell_qty = min(held, max(1, int((self.cfg.safety_buffer - cash) / price) + 1))
            return self._sell(sell_qty, price, reason="释放资金给期权")

        # Sell if options explicitly need capital
        if options_capital_needed > 0 and cash < options_capital_needed + self.cfg.safety_buffer:
            need = options_capital_needed + self.cfg.safety_buffer - cash
            sell_qty = min(held, max(1, int(need / price) + 1))
            if sell_qty > 0 and held > 0:
                return self._sell(sell_qty, price, reason=f"期权需要${options_capital_needed:.0f}")

        # Buy SGOV if idle cash is enough
        if idle >= self.cfg.min_idle_to_buy:
            buy_qty = int(idle / price)
            if buy_qty >= 1:
                return self._buy(buy_qty, price)

        return None

    def _buy(self, qty: int, price: float) -> Optional[dict]:
        logger.info("Cash Parking: 买入 %s x %d @ $%.2f", self.cfg.symbol, qty, price)
        ret, data = self.trade_ctx.place_order(
            price=round(price, 2),
            qty=qty,
            code=self.cfg.symbol,
            trd_side=TrdSide.BUY,
            order_type=OrderType.NORMAL,
            trd_env=self.trd_env,
        )
        if ret == RET_OK:
            order_id = data["order_id"].iloc[0]
            msg = (f"<b>Cash Parking 买入</b>\n"
                   f"标的: {self.cfg.symbol}\n"
                   f"数量: {qty} 股\n"
                   f"价格: ${price:.2f}\n"
                   f"金额: ${qty * price:,.0f}")
            self._notify(msg)
            return {"action": "buy", "symbol": self.cfg.symbol,
                    "qty": qty, "price": price, "order_id": order_id}
        else:
            logger.error("Cash Parking 买入失败: %s", data)
            return None

    def _sell(self, qty: int, price: float, reason: str = "") -> Optional[dict]:
        logger.info("Cash Parking: 卖出 %s x %d @ $%.2f (%s)",
                     self.cfg.symbol, qty, price, reason)
        ret, data = self.trade_ctx.place_order(
            price=round(price, 2),
            qty=qty,
            code=self.cfg.symbol,
            trd_side=TrdSide.SELL,
            order_type=OrderType.NORMAL,
            trd_env=self.trd_env,
        )
        if ret == RET_OK:
            order_id = data["order_id"].iloc[0]
            msg = (f"<b>Cash Parking 卖出</b>\n"
                   f"标的: {self.cfg.symbol}\n"
                   f"数量: {qty} 股\n"
                   f"价格: ${price:.2f}\n"
                   f"原因: {reason}")
            self._notify(msg)
            return {"action": "sell", "symbol": self.cfg.symbol,
                    "qty": qty, "price": price, "order_id": data["order_id"].iloc[0],
                    "reason": reason}
        else:
            logger.error("Cash Parking 卖出失败: %s", data)
            return None

    def _notify(self, msg: str):
        if self.notifier:
            try:
                self.notifier.send_message(msg)
            except Exception as e:
                logger.warning("Telegram 通知失败: %s", e)

    def status(self) -> dict:
        """Return current parking status for display."""
        cash = self.get_account_cash()
        held = self.get_position(self.cfg.symbol)
        price = self.get_price(self.cfg.symbol) if held > 0 else 0
        return {
            "cash": cash,
            "symbol": self.cfg.symbol,
            "shares": held,
            "value": held * price,
            "price": price,
            "idle": cash - self.cfg.safety_buffer,
        }
