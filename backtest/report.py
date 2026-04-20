import math
from typing import Any, Optional

import numpy as np
import pandas as pd


class BacktestReport:
    """Build performance metrics from a backtest `result` dict."""

    RISK_FREE_ANNUAL = 0.05

    def __init__(self, result: dict):
        self.result = result

    def _equity_returns(self, equity: list) -> np.ndarray:
        eq = np.asarray(equity, dtype=float)
        if len(eq) < 2:
            return np.array([])
        prev = eq[:-1]
        cur = eq[1:]
        mask = prev > 0
        if not np.any(mask):
            return np.array([])
        return (cur[mask] - prev[mask]) / prev[mask]

    def _years_in_backtest(self) -> float:
        data = self.result.get("data")
        total_bars = int(self.result.get("total_bars", 0) or 0)
        if data is not None and len(data) > 1 and "time_key" in getattr(data, "columns", []):
            t0 = pd.Timestamp(data.iloc[0]["time_key"])
            t1 = pd.Timestamp(data.iloc[-1]["time_key"])
            sec = (t1 - t0).total_seconds()
            if sec > 0:
                return sec / (365.25 * 24 * 3600)
        return max(total_bars - 1, 1) / 252.0

    def _periods_per_year(self) -> float:
        years = self._years_in_backtest()
        total_bars = int(self.result.get("total_bars", 0) or 0)
        n_ret = max(total_bars - 1, 1)
        if years > 0:
            return n_ret / years
        return 252.0

    def _bar_index_for_trade_time(
        self, data: Optional[pd.DataFrame], t: Any, total_bars: int
    ) -> Optional[int]:
        if isinstance(t, (int, np.integer)):
            i = int(t)
            if 0 <= i < total_bars:
                return i
            return None
        if data is None or "time_key" not in data.columns:
            return None
        hits = np.flatnonzero((data["time_key"] == t).to_numpy())
        if len(hits):
            return int(hits[0])
        return None

    def _exposure_avg_holding(
        self, trades: list, total_bars: int, data: Optional[pd.DataFrame]
    ) -> tuple[float, float]:
        in_market = np.zeros(max(total_bars, 0), dtype=bool)
        hold_days: list[float] = []
        open_from: Optional[int] = None
        buy_time: Any = None

        for t in trades:
            if t.get("type") == "BUY":
                bi = self._bar_index_for_trade_time(data, t.get("time"), total_bars)
                if bi is None:
                    continue
                open_from = bi
                buy_time = t.get("time")
            elif t.get("type") == "SELL" and open_from is not None:
                si = self._bar_index_for_trade_time(data, t.get("time"), total_bars)
                if si is None:
                    continue
                lo, hi = (open_from, si) if open_from <= si else (si, open_from)
                in_market[lo : hi + 1] = True
                if data is not None and "time_key" in data.columns and buy_time is not None:
                    t0 = pd.Timestamp(buy_time)
                    t1 = pd.Timestamp(t["time"])
                    hold_days.append((t1 - t0).total_seconds() / 86400.0)
                else:
                    hold_days.append(float(hi - lo + 1))
                open_from = None
                buy_time = None

        if open_from is not None and total_bars > 0:
            in_market[open_from : total_bars] = True

        exposure_pct = float(np.mean(in_market) * 100) if total_bars > 0 else 0.0
        avg_hold = float(np.mean(hold_days)) if hold_days else 0.0
        return exposure_pct, avg_hold

    def _max_drawdown_fraction(self, equity: list) -> float:
        if not equity:
            return 0.0
        peak = float(equity[0])
        max_dd = 0.0
        for val in equity:
            v = float(val)
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def _max_consecutive_wins_losses(self, trades: list) -> tuple[int, int]:
        sells = [t for t in trades if t.get("type") == "SELL" and "pnl" in t]
        max_w, max_l = 0, 0
        cw, cl = 0, 0
        for t in sells:
            pnl = float(t.get("pnl", 0))
            if pnl > 0:
                cw += 1
                cl = 0
                max_w = max(max_w, cw)
            else:
                cl += 1
                cw = 0
                max_l = max(max_l, cl)
        return max_l, max_w

    def _monthly_returns_pct(self, equity: list, data: pd.DataFrame, initial: float) -> dict[str, float]:
        if len(equity) != len(data) or "time_key" not in data.columns:
            return {}
        idx = pd.to_datetime(data["time_key"], errors="coerce")
        s = pd.Series(equity, index=idx)
        s = s[~s.index.duplicated(keep="last")]
        if s.empty:
            return {}
        month_end = s.resample("ME").last()
        prev = month_end.shift(1)
        if not prev.empty and pd.isna(prev.iloc[0]):
            prev = prev.copy()
            prev.iloc[0] = initial
        rets = (month_end / prev - 1.0) * 100.0
        out: dict[str, float] = {}
        for ts, v in rets.items():
            if pd.notna(v):
                out[ts.strftime("%Y-%m")] = round(float(v), 4)
        return out

    def _buy_hold(self, initial: float, data: pd.DataFrame) -> dict[str, float]:
        first = float(data.iloc[0]["close"])
        last = float(data.iloc[-1]["close"])
        if first <= 0:
            return {"benchmark_final_capital": initial, "benchmark_total_return_pct": 0.0}
        mult = last / first
        return {
            "benchmark_final_capital": round(initial * mult, 2),
            "benchmark_total_return_pct": round((mult - 1.0) * 100.0, 4),
        }

    def summary(self) -> dict:
        initial = float(self.result["initial_capital"])
        final = float(self.result["final_capital"])
        trades = self.result["trades"]
        equity = self.result["equity_curve"]
        total_bars = int(self.result.get("total_bars", len(equity)) or 0)
        data = self.result.get("data")

        total_return = final - initial
        total_return_pct = (total_return / initial) * 100 if initial else 0.0

        sell_trades = [t for t in trades if t.get("type") == "SELL"]
        wins = [t for t in sell_trades if float(t.get("pnl", 0)) > 0]
        losses = [t for t in sell_trades if float(t.get("pnl", 0)) <= 0]

        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0.0
        avg_win = float(np.mean([float(t["pnl"]) for t in wins])) if wins else 0.0
        avg_loss = abs(float(np.mean([float(t["pnl"]) for t in losses]))) if losses else 0.0

        sum_wins = float(sum(float(t["pnl"]) for t in wins)) if wins else 0.0
        sum_losses = float(sum(float(t["pnl"]) for t in losses)) if losses else 0.0
        if sum_losses < 0:
            profit_factor = sum_wins / abs(sum_losses)
        elif sum_wins > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        max_drawdown_pct = self._max_drawdown_fraction(equity) * 100.0
        max_dd_frac = max_drawdown_pct / 100.0

        total_commission = float(sum(float(t.get("commission", 0)) for t in trades))

        years = self._years_in_backtest()
        if years > 0 and initial > 0 and final > 0:
            cagr_pct = ((final / initial) ** (1.0 / years) - 1.0) * 100.0
        else:
            cagr_pct = 0.0

        if max_dd_frac > 1e-12 and years > 0:
            calmar_ratio = (cagr_pct / 100.0) / max_dd_frac
        elif max_dd_frac <= 1e-12 and cagr_pct > 0:
            calmar_ratio = float("inf")
        else:
            calmar_ratio = 0.0

        rets = self._equity_returns(equity)
        ppy = self._periods_per_year()
        rf_per = (1.0 + self.RISK_FREE_ANNUAL) ** (1.0 / ppy) - 1.0 if ppy > 0 else 0.0

        if len(rets) > 1:
            excess = rets - rf_per
            std = float(np.std(rets, ddof=1))
            if std > 1e-12:
                sharpe_ratio = float(np.mean(excess) / std * math.sqrt(ppy))
            else:
                sharpe_ratio = 0.0 if abs(float(np.mean(excess))) < 1e-12 else float("nan")

            downside = np.minimum(0.0, rets - rf_per)
            ddev = float(np.sqrt(np.mean(downside**2)))
            if ddev > 1e-12:
                sortino_ratio = float(np.mean(excess) / ddev * math.sqrt(ppy))
            else:
                sortino_ratio = 0.0 if abs(float(np.mean(excess))) < 1e-12 else float("inf")
        elif len(rets) == 1:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        exposure_pct, avg_holding_days = self._exposure_avg_holding(trades, total_bars, data)
        max_consecutive_losses, max_consecutive_wins = self._max_consecutive_wins_losses(trades)

        monthly_returns: dict[str, float] = {}
        if data is not None and isinstance(data, pd.DataFrame):
            monthly_returns = self._monthly_returns_pct(equity, data, initial)

        benchmark: dict[str, Any] = {}
        if data is not None and isinstance(data, pd.DataFrame) and len(data) > 0 and "close" in data.columns:
            bh = self._buy_hold(initial, data)
            benchmark = {
                "benchmark_final_capital": bh["benchmark_final_capital"],
                "benchmark_total_return_pct": bh["benchmark_total_return_pct"],
                "strategy_vs_benchmark_pct": round(total_return_pct - bh["benchmark_total_return_pct"], 4),
            }

        out: dict[str, Any] = {
            "initial_capital": initial,
            "final_capital": round(final, 2),
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": len(sell_trades),
            "win_rate_pct": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else profit_factor,
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_commission": round(total_commission, 2),
            "sharpe_ratio": round(sharpe_ratio, 4) if math.isfinite(sharpe_ratio) else sharpe_ratio,
            "sortino_ratio": round(sortino_ratio, 4) if math.isfinite(sortino_ratio) else sortino_ratio,
            "calmar_ratio": round(calmar_ratio, 4) if math.isfinite(calmar_ratio) else calmar_ratio,
            "cagr_pct": round(cagr_pct, 4),
            "exposure_pct": round(exposure_pct, 4),
            "avg_holding_period_days": round(avg_holding_days, 4),
            "max_consecutive_losses": int(max_consecutive_losses),
            "max_consecutive_wins": int(max_consecutive_wins),
            "monthly_returns": monthly_returns,
        }
        out.update(benchmark)
        return out

    def print_report(self) -> str:
        s = self.summary()
        lines = [
            "=" * 58,
            "BACKTEST REPORT",
            "=" * 58,
            f"Initial Capital:           ${s['initial_capital']:,.2f}",
            f"Final Capital:             ${s['final_capital']:,.2f}",
            f"Total Return:              ${s['total_return']:+.2f} ({s['total_return_pct']:+.2f}%)",
            f"CAGR:                      {s['cagr_pct']:.4f}%",
            f"Max Drawdown:              {s['max_drawdown_pct']:.2f}%",
            f"Sharpe (rf {self.RISK_FREE_ANNUAL * 100:.0f}% p.a.):   {s['sharpe_ratio']}",
            f"Sortino:                   {s['sortino_ratio']}",
            f"Calmar:                    {s['calmar_ratio']}",
            f"Exposure (time in mkt):    {s['exposure_pct']:.2f}%",
            f"Avg holding (days):        {s['avg_holding_period_days']:.2f}",
            f"Max consecutive losses:    {s['max_consecutive_losses']}",
            f"Max consecutive wins:        {s['max_consecutive_wins']}",
            "-" * 58,
            f"Total Trades (round-trips): {s['total_trades']}",
            f"Win Rate:                  {s['win_rate_pct']:.2f}%",
            f"Avg Win:                   ${s['avg_win']:,.2f}",
            f"Avg Loss:                  ${s['avg_loss']:,.2f}",
            f"Profit Factor:             {s['profit_factor']}",
            f"Total Commission:          ${s['total_commission']:,.2f}",
        ]
        if s.get("benchmark_final_capital") is not None:
            lines.extend(
                [
                    "-" * 58,
                    "BUY & HOLD BENCHMARK",
                    f"Benchmark Final:           ${s['benchmark_final_capital']:,.2f}",
                    f"Benchmark Return:          {s['benchmark_total_return_pct']:+.4f}%",
                    f"Strategy vs Benchmark:     {s['strategy_vs_benchmark_pct']:+.4f}% (return diff)",
                ]
            )
        lines.append("-" * 58)
        lines.append("MONTHLY RETURNS %")
        mr = s.get("monthly_returns") or {}
        if mr:
            for k in sorted(mr.keys()):
                lines.append(f"  {k}:  {mr[k]:+.4f}%")
        else:
            lines.append("  (n/a — need aligned `data` time_key + equity_curve)")
        lines.append("=" * 58)
        report = "\n".join(lines)
        print(report)
        return report
