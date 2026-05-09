"""
Futu Paper Trading Adapter — 通过 Futu SIMULATE 环境执行交易

与 PaperTrader 的因子选股结果对比：
  - PaperTrader: yfinance 收盘价虚拟交易 (无滑点)
  - FutuPaperTrader: Futu 模拟环境真实委托 (有滑点和成交延迟)
"""
from __future__ import annotations

import time
from datetime import date
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("futu_paper")


class FutuPaperTrader:
    """Execute paper trades via Futu OpenD SIMULATE environment."""

    def __init__(self, host: str = "127.0.0.1", port: int = 11111):
        self.host = host
        self.port = port
        self._trade_ctx = None
        self._quote_ctx = None

    def connect(self) -> bool:
        try:
            from futu import OpenSecTradeContext, OpenQuoteContext
            self._trade_ctx = OpenSecTradeContext(host=self.host, port=self.port)
            self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)
            logger.info("Futu paper trading connected (SIMULATE)")
            return True
        except Exception as e:
            logger.error(f"Futu connection failed: {e}")
            return False

    def disconnect(self):
        if self._trade_ctx:
            self._trade_ctx.close()
        if self._quote_ctx:
            self._quote_ctx.close()

    def get_account_info(self) -> Optional[dict]:
        from futu import RET_OK, TrdEnv
        ret, data = self._trade_ctx.accinfo_query(trd_env=TrdEnv.SIMULATE)
        if ret != RET_OK or data is None or data.empty:
            return None
        row = data.iloc[0]
        return {
            "total_assets": float(row.get("total_assets", 0)),
            "cash": float(row.get("cash", 0)),
            "market_val": float(row.get("market_val", 0)),
            "available_funds": float(row.get("available_funds", 0)),
        }

    def get_positions(self) -> list[dict]:
        from futu import RET_OK, TrdEnv
        ret, data = self._trade_ctx.position_list_query(trd_env=TrdEnv.SIMULATE)
        if ret != RET_OK or data is None or data.empty:
            return []
        positions = []
        for _, row in data.iterrows():
            qty = int(row.get("qty", 0))
            if qty == 0:
                continue
            positions.append({
                "code": str(row.get("code", "")),
                "qty": qty,
                "cost_price": float(row.get("cost_price", 0)),
                "market_val": float(row.get("market_val", 0)),
                "pl_val": float(row.get("pl_val", 0)),
                "pl_ratio": float(row.get("pl_ratio", 0)),
            })
        return positions

    def place_order(self, symbol: str, qty: int, price: float,
                    side: str = "BUY") -> Optional[str]:
        """Place a SIMULATE order. Returns order_id or None."""
        from futu import RET_OK, TrdSide, OrderType, TrdEnv

        futu_side = TrdSide.BUY if side == "BUY" else TrdSide.SELL
        code = symbol if symbol.startswith("US.") else f"US.{symbol}"

        time.sleep(0.5)
        ret, data = self._trade_ctx.place_order(
            price=price, qty=qty, code=code,
            trd_side=futu_side, order_type=OrderType.NORMAL,
            trd_env=TrdEnv.SIMULATE,
        )
        if ret == RET_OK and data is not None and len(data) > 0:
            order_id = str(data.iloc[0].get("order_id", ""))
            logger.info(f"[FUTU SIM] {side} {qty}x {code} @ ${price:.2f} "
                        f"order_id={order_id}")
            return order_id
        logger.error(f"[FUTU SIM] Order failed: {data}")
        return None

    def execute_rebalance(self, target_symbols: list[str],
                          capital: float = 10000.0) -> dict:
        """Execute a full rebalance in Futu SIMULATE.

        1. Sell positions not in target
        2. Buy equal-weight into target symbols
        Returns summary dict.
        """
        from futu import RET_OK

        current_positions = self.get_positions()
        current_codes = {p["code"] for p in current_positions}
        target_codes = {
            s if s.startswith("US.") else f"US.{s}" for s in target_symbols
        }

        sells = []
        for pos in current_positions:
            code = pos["code"]
            if code not in target_codes and pos["qty"] > 0:
                time.sleep(0.5)
                ret, snap = self._quote_ctx.get_market_snapshot([code])
                if ret == RET_OK and snap is not None and len(snap) > 0:
                    price = float(snap.iloc[0].get("bid_price",
                                  snap.iloc[0].get("last_price", pos["cost_price"])))
                else:
                    price = pos["cost_price"]

                if price > 0:
                    oid = self.place_order(code, pos["qty"], price, side="SELL")
                    if oid:
                        sells.append({"code": code, "qty": pos["qty"],
                                      "price": price})

        # Compute available capital from account
        time.sleep(1)
        acct = self.get_account_info()
        available = acct["available_funds"] if acct else capital

        # Buy into new targets
        to_buy = [c for c in target_codes if c not in current_codes or
                  c in {s["code"] for s in sells}]
        buys = []
        if to_buy:
            per_stock = available / len(to_buy)
            for code in to_buy:
                time.sleep(0.5)
                ret, snap = self._quote_ctx.get_market_snapshot([code])
                if ret != RET_OK or snap is None or snap.empty:
                    continue
                price = float(snap.iloc[0].get("ask_price",
                              snap.iloc[0].get("last_price", 0)))
                if price <= 0:
                    continue
                qty = max(int(per_stock / price), 1)
                oid = self.place_order(code, qty, price, side="BUY")
                if oid:
                    buys.append({"code": code, "qty": qty, "price": price})

        summary = {
            "date": date.today().isoformat(),
            "sells": len(sells),
            "buys": len(buys),
            "available_before": available,
        }
        logger.info(f"[FUTU SIM] Rebalance: {len(sells)} sells, {len(buys)} buys")
        return summary

    def compare_with_paper(self, paper_holdings: list[dict]) -> dict:
        """Compare Futu SIMULATE positions with PaperTrader holdings.

        Returns slippage analysis.
        """
        futu_positions = self.get_positions()
        futu_map = {}
        for p in futu_positions:
            ticker = p["code"].replace("US.", "")
            futu_map[ticker] = p

        slippage = []
        for h in paper_holdings:
            sym = h["symbol"]
            if sym in futu_map:
                fp = futu_map[sym]
                cost_diff = fp["cost_price"] - h["cost_price"]
                slippage.append({
                    "symbol": sym,
                    "paper_cost": h["cost_price"],
                    "futu_cost": fp["cost_price"],
                    "slippage": cost_diff,
                    "slippage_bps": (cost_diff / h["cost_price"] * 10000
                                     if h["cost_price"] > 0 else 0),
                })

        avg_slip = (sum(s["slippage_bps"] for s in slippage) / len(slippage)
                    if slippage else 0)
        return {
            "n_compared": len(slippage),
            "avg_slippage_bps": round(avg_slip, 2),
            "details": slippage,
        }
