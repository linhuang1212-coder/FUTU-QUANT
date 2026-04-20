import numpy as np


class BacktestReport:
    def __init__(self, result: dict):
        self.result = result

    def summary(self) -> dict:
        initial = self.result["initial_capital"]
        final = self.result["final_capital"]
        trades = self.result["trades"]
        equity = self.result["equity_curve"]

        total_return = final - initial
        total_return_pct = (total_return / initial) * 100

        sell_trades = [t for t in trades if t["type"] == "SELL"]
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

        max_drawdown_pct = 0.0
        if equity:
            peak = equity[0]
            for val in equity:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak * 100
                if dd > max_drawdown_pct:
                    max_drawdown_pct = dd

        total_commission = sum(t.get("commission", 0) for t in trades)

        return {
            "initial_capital": initial,
            "final_capital": round(final, 2),
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": len(sell_trades),
            "win_rate_pct": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_commission": round(total_commission, 2),
        }

    def print_report(self) -> str:
        s = self.summary()
        lines = [
            "=" * 50,
            "BACKTEST REPORT",
            "=" * 50,
            f"Initial Capital:  ${s['initial_capital']:,.2f}",
            f"Final Capital:    ${s['final_capital']:,.2f}",
            f"Total Return:     ${s['total_return']:+.2f} ({s['total_return_pct']:+.2f}%)",
            f"Max Drawdown:     {s['max_drawdown_pct']:.2f}%",
            f"Total Trades:     {s['total_trades']}",
            f"Win Rate:         {s['win_rate_pct']:.2f}%",
            f"Avg Win:          ${s['avg_win']:,.2f}",
            f"Avg Loss:         ${s['avg_loss']:,.2f}",
            f"Profit Factor:    {s['profit_factor']:.2f}",
            f"Total Commission: ${s['total_commission']:,.2f}",
            "=" * 50,
        ]
        report = "\n".join(lines)
        print(report)
        return report
