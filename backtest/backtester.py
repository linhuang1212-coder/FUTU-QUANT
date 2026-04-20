import pandas as pd
from strategy.base import BaseStrategy, SignalDirection
from utils.logger import setup_logger

logger = setup_logger("backtester")


class Backtester:
    def __init__(self, initial_capital: float = 3000, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(self, strategy: BaseStrategy, symbol: str, data: pd.DataFrame) -> dict:
        capital = self.initial_capital
        position = 0
        avg_entry = 0.0
        trades = []
        equity_curve = []

        for i in range(len(data)):
            bar = data.iloc[:i + 1]
            current_price = data.iloc[i]["close"]
            signal = strategy.on_bar(symbol, bar)

            if signal is not None:
                if signal.direction == SignalDirection.BUY and position == 0:
                    slippage = current_price * self.slippage_pct
                    buy_price = current_price + slippage
                    max_shares = int(capital * 0.4 / buy_price)
                    if max_shares > 0:
                        commission = buy_price * max_shares * self.commission_pct
                        cost = buy_price * max_shares + commission
                        if cost <= capital:
                            position = max_shares
                            avg_entry = buy_price
                            capital -= cost
                            trades.append({
                                "type": "BUY",
                                "price": buy_price,
                                "quantity": max_shares,
                                "commission": commission,
                                "time": data.iloc[i].get("time_key", i),
                            })

                elif signal.direction == SignalDirection.SELL and position > 0:
                    slippage = current_price * self.slippage_pct
                    sell_price = current_price - slippage
                    commission = sell_price * position * self.commission_pct
                    revenue = sell_price * position - commission
                    pnl = (sell_price - avg_entry) * position - commission
                    capital += revenue
                    trades.append({
                        "type": "SELL",
                        "price": sell_price,
                        "quantity": position,
                        "commission": commission,
                        "pnl": pnl,
                        "time": data.iloc[i].get("time_key", i),
                    })
                    position = 0
                    avg_entry = 0.0

            portfolio_value = capital + position * current_price
            equity_curve.append(portfolio_value)

        final_value = capital + position * data.iloc[-1]["close"]

        return {
            "initial_capital": self.initial_capital,
            "final_capital": final_value,
            "trades": trades,
            "equity_curve": equity_curve,
            "total_bars": len(data),
        }
