import numpy as np
import pandas as pd
from strategy.base import BaseStrategy, SignalDirection
from utils.logger import setup_logger

logger = setup_logger("backtester")


def _compute_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        pc = close[i - 1]
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - pc),
            abs(low[i] - pc),
        )
    return tr


def _wilder_atr(tr: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(tr)
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _allocation_from_strength(strength: float) -> float:
    """Fraction of capital to allocate to new exposure (long or short notional).

    For small accounts ($3,000), full deployment maximises capital efficiency.
    A 5% cash reserve is kept for commission/slippage headroom.
    """
    return 0.95


class Backtester:
    def __init__(self, initial_capital: float = 3000, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pd.DataFrame,
        lookback: int = 50,
        use_atr_stop: bool = False,
        allow_short: bool = False,
    ) -> dict:
        capital = self.initial_capital
        position = 0
        avg_entry = 0.0
        stop_price: float | None = None
        trades: list[dict] = []
        equity_curve: list[float] = []

        n = len(data)
        if n == 0:
            return {
                "initial_capital": self.initial_capital,
                "final_capital": self.initial_capital,
                "trades": [],
                "equity_curve": [],
                "total_bars": 0,
                "data": data,
            }

        close = data["close"].to_numpy(dtype=float, copy=False)
        high = data["high"].to_numpy(dtype=float, copy=False) if "high" in data.columns else close.copy()
        low = data["low"].to_numpy(dtype=float, copy=False) if "low" in data.columns else close.copy()

        atr: np.ndarray | None = None
        if use_atr_stop:
            tr = _compute_true_range(high, low, close)
            atr = _wilder_atr(tr, period=14)

        lb = max(1, int(lookback))

        for i in range(n):
            current_price = float(close[i])
            bar_high = float(high[i])
            bar_low = float(low[i])

            # Optional ATR stop: evaluate before new signals on this bar
            if use_atr_stop and stop_price is not None:
                if position > 0 and bar_low <= stop_price:
                    fill = float(stop_price)
                    commission = fill * position * self.commission_pct
                    revenue = fill * position - commission
                    pnl = (fill - avg_entry) * position - commission
                    capital += revenue
                    trades.append(
                        {
                            "type": "SELL",
                            "price": fill,
                            "quantity": position,
                            "commission": commission,
                            "pnl": pnl,
                            "time": data.iloc[i].get("time_key", i),
                            "reason": "atr_stop",
                        }
                    )
                    position = 0
                    avg_entry = 0.0
                    stop_price = None
                elif position < 0 and bar_high >= stop_price:
                    fill = float(stop_price)
                    qty = abs(position)
                    commission = fill * qty * self.commission_pct
                    cost = fill * qty + commission
                    pnl = (avg_entry - fill) * qty - commission
                    capital -= cost
                    trades.append(
                        {
                            "type": "BUY",
                            "price": fill,
                            "quantity": qty,
                            "commission": commission,
                            "pnl": pnl,
                            "time": data.iloc[i].get("time_key", i),
                            "reason": "atr_stop",
                        }
                    )
                    position = 0
                    avg_entry = 0.0
                    stop_price = None

            start = max(0, i - lb + 1)
            window = data.iloc[start : i + 1]
            signal = strategy.on_bar(symbol, window)

            if signal is not None:
                alloc = _allocation_from_strength(float(signal.strength))

                if signal.direction == SignalDirection.BUY and position == 0:
                    slippage = current_price * self.slippage_pct
                    buy_price = current_price + slippage
                    max_shares = int(capital * alloc / buy_price)
                    if max_shares > 0:
                        commission = buy_price * max_shares * self.commission_pct
                        cost = buy_price * max_shares + commission
                        if cost <= capital:
                            position = max_shares
                            avg_entry = buy_price
                            capital -= cost
                            entry_atr = float(atr[i]) if use_atr_stop and atr is not None and np.isfinite(atr[i]) else float("nan")
                            stop_price = (
                                avg_entry - 2.0 * entry_atr
                                if use_atr_stop and np.isfinite(entry_atr)
                                else None
                            )
                            trades.append(
                                {
                                    "type": "BUY",
                                    "price": buy_price,
                                    "quantity": max_shares,
                                    "commission": commission,
                                    "time": data.iloc[i].get("time_key", i),
                                }
                            )

                elif signal.direction == SignalDirection.BUY and position < 0:
                    slippage = current_price * self.slippage_pct
                    buy_price = current_price + slippage
                    qty = abs(position)
                    commission = buy_price * qty * self.commission_pct
                    cost = buy_price * qty + commission
                    pnl = (avg_entry - buy_price) * qty - commission
                    if cost <= capital:
                        capital -= cost
                        trades.append(
                            {
                                "type": "BUY",
                                "price": buy_price,
                                "quantity": qty,
                                "commission": commission,
                                "pnl": pnl,
                                "time": data.iloc[i].get("time_key", i),
                            }
                        )
                        position = 0
                        avg_entry = 0.0
                        stop_price = None

                elif signal.direction == SignalDirection.SELL and position > 0:
                    slippage = current_price * self.slippage_pct
                    sell_price = current_price - slippage
                    commission = sell_price * position * self.commission_pct
                    revenue = sell_price * position - commission
                    pnl = (sell_price - avg_entry) * position - commission
                    capital += revenue
                    trades.append(
                        {
                            "type": "SELL",
                            "price": sell_price,
                            "quantity": position,
                            "commission": commission,
                            "pnl": pnl,
                            "time": data.iloc[i].get("time_key", i),
                        }
                    )
                    position = 0
                    avg_entry = 0.0
                    stop_price = None

                elif allow_short and signal.direction == SignalDirection.SELL and position == 0:
                    slippage = current_price * self.slippage_pct
                    sell_price = current_price - slippage
                    max_shares = int(capital * alloc / sell_price)
                    if max_shares > 0:
                        commission = sell_price * max_shares * self.commission_pct
                        proceeds = sell_price * max_shares - commission
                        if proceeds > 0:
                            position = -max_shares
                            avg_entry = sell_price
                            capital += proceeds
                            entry_atr = (
                                float(atr[i]) if use_atr_stop and atr is not None and np.isfinite(atr[i]) else float("nan")
                            )
                            stop_price = (
                                avg_entry + 2.0 * entry_atr
                                if use_atr_stop and np.isfinite(entry_atr)
                                else None
                            )
                            trades.append(
                                {
                                    "type": "SHORT",
                                    "price": sell_price,
                                    "quantity": max_shares,
                                    "commission": commission,
                                    "time": data.iloc[i].get("time_key", i),
                                }
                            )

            portfolio_value = capital + position * current_price
            equity_curve.append(portfolio_value)

        final_value = capital + position * float(close[-1])

        return {
            "initial_capital": self.initial_capital,
            "final_capital": final_value,
            "trades": trades,
            "equity_curve": equity_curve,
            "total_bars": n,
            "data": data,
        }
