"""Options backtest engine with synthetic IV / Black-Scholes pricing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

from options.pricer import bs_price, implied_vol, greeks, bb_width, compute_ivr
from utils.logger import setup_logger

logger = setup_logger("options.backtest")


@dataclass
class BacktestTrade:
    strategy: str
    underlying: str
    direction: str  # "CALL" / "PUT" / "STRADDLE" / "SPREAD"
    entry_date: str
    exit_date: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    strike: float = 0.0
    contracts: int = 1
    premium_paid: float = 0.0
    pnl: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    strategy: str
    trades: list[BacktestTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0

    def compute_stats(self):
        if not self.trades:
            return
        pnls = [t.pnl for t in self.trades]
        self.total_trades = len(pnls)
        self.total_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        self.win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = np.mean(losses) if losses else 0

        if len(pnls) > 1 and np.std(pnls) > 0:
            self.sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))

        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        dd = cumulative - peak
        self.max_drawdown = float(np.min(dd)) if len(dd) > 0 else 0


def _synth_iv(hist_vol: float, days_to_expiry: int) -> float:
    """Synthesize IV from historical vol with term structure adjustment."""
    base = hist_vol * 1.2
    if days_to_expiry <= 1:
        return base * 1.5  # 0DTE elevated IV
    elif days_to_expiry <= 7:
        return base * 1.2
    return base


def _hist_vol(closes: np.ndarray, window: int = 20) -> float:
    if len(closes) < window + 1:
        return 0.3
    rets = np.diff(np.log(closes[-window - 1:]))
    return float(np.std(rets) * np.sqrt(252))


class OptionsBacktester:
    """Synthetic options backtester using historical price data."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def backtest_orb(self, df: pd.DataFrame, config: dict) -> BacktestResult:
        """Backtest 0DTE ORB strategy on intraday 5-min data."""
        result = BacktestResult(strategy="orb_0dte")
        stop_loss_pct = config.get("stop_loss_pct", 0.50)
        target_pct = config.get("target_pct", 1.00)
        max_premium = config.get("max_premium", 300)

        if "time_key" not in df.columns or len(df) < 50:
            logger.warning("ORB backtest requires intraday data with time_key column")
            return result

        df = df.copy()
        df["date"] = pd.to_datetime(df["time_key"]).dt.date

        for date, day_df in df.groupby("date"):
            if len(day_df) < 12:
                continue

            # ORB: first 6 bars (30 min of 5-min data)
            orb_bars = day_df.head(6)
            orb_h = orb_bars["high"].max()
            orb_l = orb_bars["low"].min()
            rest = day_df.iloc[6:]

            triggered = False
            for idx, bar in rest.iterrows():
                if triggered:
                    break
                close = bar["close"]
                if close > orb_h:
                    direction = "CALL"
                elif close < orb_l:
                    direction = "PUT"
                else:
                    continue

                triggered = True
                spot = close
                strike = round(spot)
                vol = _hist_vol(day_df["close"].values[:day_df.index.get_loc(idx)])
                iv = _synth_iv(vol if vol > 0 else 0.25, 0)
                T = max(1 / (252 * 78), 0.0001)  # fraction of day remaining

                entry_price = bs_price(spot, strike, T, self.r, iv, direction)
                if entry_price <= 0.01:
                    continue

                contracts = max(1, int(max_premium / (entry_price * 100)))
                premium = entry_price * contracts * 100

                # Simulate exit on remaining bars
                exit_price = entry_price
                reason = "EOD"
                remaining_bars = day_df.loc[day_df.index > idx]

                for _, ebar in remaining_bars.iterrows():
                    T_now = max(T * 0.5, 0.0001)
                    opt_price = bs_price(ebar["close"], strike, T_now, self.r, iv, direction)
                    pnl_pct = (opt_price - entry_price) / entry_price

                    if pnl_pct <= -stop_loss_pct:
                        exit_price = entry_price * (1 - stop_loss_pct)
                        reason = "Stop loss"
                        break
                    elif pnl_pct >= target_pct:
                        exit_price = entry_price * (1 + target_pct)
                        reason = "Take profit"
                        break
                    exit_price = opt_price

                pnl = (exit_price - entry_price) * contracts * 100
                trade = BacktestTrade(
                    strategy="orb_0dte", underlying=str(date),
                    direction=direction, entry_date=str(date),
                    exit_date=str(date), entry_price=entry_price,
                    exit_price=exit_price, strike=strike,
                    contracts=contracts, premium_paid=premium, pnl=pnl,
                    reason=reason,
                )
                result.trades.append(trade)

        result.compute_stats()
        return result

    def backtest_credit_spread(self, df: pd.DataFrame, config: dict) -> BacktestResult:
        """Backtest credit spread on daily data."""
        result = BacktestResult(strategy="credit_spread")
        min_ivr = config.get("min_ivr", 60)
        profit_take_pct = config.get("profit_take_pct", 0.50)
        stop_loss_pct = config.get("stop_loss_pct", 2.00)
        max_hold = config.get("max_holding_days", 21)
        spread_width = config.get("spread_width", 5.0)

        if len(df) < 60:
            return result

        closes = df["close"].values

        # Compute rolling vol for IVR
        rets = np.diff(np.log(closes))
        for i in range(252, len(closes)):
            window_vols = []
            for j in range(20, min(i, 252)):
                wv = float(np.std(rets[j - 20:j]) * np.sqrt(252))
                window_vols.append(wv)
            if not window_vols:
                continue
            current_vol = float(np.std(rets[i - 20:i]) * np.sqrt(252))
            ivr = compute_ivr(current_vol, window_vols)

            if ivr < min_ivr:
                continue

            spot = closes[i]
            short_strike = round(spot * 0.95)
            long_strike = short_strike - spread_width
            T = max_hold / 252
            iv = _synth_iv(current_vol, max_hold)

            short_put_price = bs_price(spot, short_strike, T, self.r, iv, "PUT")
            long_put_price = bs_price(spot, long_strike, T, self.r, iv, "PUT")
            credit = short_put_price - long_put_price
            if credit <= 0.05:
                continue

            max_loss_val = (short_strike - long_strike - credit) * 100

            # Simulate holding period
            exit_pnl = credit * 100
            reason = "Expiry"
            for d in range(1, min(max_hold + 1, len(closes) - i)):
                future_spot = closes[i + d]
                T_rem = max((max_hold - d) / 252, 0.001)
                short_now = bs_price(future_spot, short_strike, T_rem, self.r, iv, "PUT")
                long_now = bs_price(future_spot, long_strike, T_rem, self.r, iv, "PUT")
                spread_now = short_now - long_now
                pnl = (credit - spread_now) * 100

                if pnl >= credit * 100 * profit_take_pct:
                    exit_pnl = pnl
                    reason = "Take profit"
                    break
                if pnl <= -max_loss_val * stop_loss_pct:
                    exit_pnl = pnl
                    reason = "Stop loss"
                    break
                exit_pnl = pnl

            trade = BacktestTrade(
                strategy="credit_spread",
                underlying=df["code"].iloc[0] if "code" in df.columns else "unknown",
                direction="SPREAD", entry_date=str(df.index[i]) if hasattr(df.index, '__getitem__') else str(i),
                entry_price=credit, exit_price=credit - exit_pnl / 100,
                strike=short_strike, premium_paid=0,
                pnl=exit_pnl, reason=reason,
            )
            result.trades.append(trade)

        result.compute_stats()
        return result

    def backtest_straddle(self, df: pd.DataFrame, config: dict) -> BacktestResult:
        """Backtest straddle squeeze on daily data."""
        result = BacktestResult(strategy="straddle")
        lookback = config.get("bb_width_lookback", 126)
        pct_threshold = config.get("bb_width_percentile", 5)
        max_hold = config.get("max_holding_days", 14)

        if len(df) < lookback + 30:
            return result

        closes = df["close"].values
        widths = bb_width(closes)

        for i in range(lookback + 20, len(closes)):
            valid_w = widths[i - lookback:i]
            valid_w = valid_w[~np.isnan(valid_w)]
            if len(valid_w) < lookback // 2:
                continue
            current_w = widths[i]
            if np.isnan(current_w):
                continue
            threshold = float(np.percentile(valid_w, pct_threshold))
            if current_w > threshold:
                continue

            spot = closes[i]
            strike = round(spot)
            vol = _hist_vol(closes[:i + 1])
            T = max_hold / 252
            iv = _synth_iv(vol, max_hold)

            call_price = bs_price(spot, strike, T, self.r, iv, "CALL")
            put_price = bs_price(spot, strike, T, self.r, iv, "PUT")
            total_premium = call_price + put_price
            if total_premium <= 0.1:
                continue

            # Simulate
            best_pnl = 0
            reason = "Max hold"
            for d in range(1, min(max_hold + 1, len(closes) - i)):
                future_spot = closes[i + d]
                T_rem = max((max_hold - d) / 252, 0.001)
                call_now = bs_price(future_spot, strike, T_rem, self.r, iv, "CALL")
                put_now = bs_price(future_spot, strike, T_rem, self.r, iv, "PUT")
                pnl = (call_now + put_now - total_premium) * 100

                if abs(future_spot - spot) / spot > 0.03:
                    best_pnl = pnl
                    reason = "Breakout"
                    break
                if pnl <= -total_premium * 100 * 0.5:
                    best_pnl = pnl
                    reason = "Stop loss"
                    break
                best_pnl = pnl

            trade = BacktestTrade(
                strategy="straddle",
                underlying=df["code"].iloc[0] if "code" in df.columns else "unknown",
                direction="STRADDLE",
                entry_date=str(i), entry_price=total_premium,
                exit_price=total_premium + best_pnl / 100,
                strike=strike, premium_paid=total_premium * 100,
                pnl=best_pnl, reason=reason,
            )
            result.trades.append(trade)

        result.compute_stats()
        return result


def print_backtest_report(result: BacktestResult, n_trials: int = 1):
    """Print formatted backtest results with statistical validation."""
    print(f"\n{'=' * 60}")
    print(f"  期权回测报告: {result.strategy}")
    print(f"{'=' * 60}")
    print(f"  总交易数:     {result.total_trades}")
    print(f"  总盈亏:       ${result.total_pnl:,.2f}")
    print(f"  胜率:         {result.win_rate:.1f}%")
    print(f"  平均盈利:     ${result.avg_win:,.2f}")
    print(f"  平均亏损:     ${result.avg_loss:,.2f}")
    print(f"  Sharpe:       {result.sharpe:.2f}")
    print(f"  最大回撤:     ${result.max_drawdown:,.2f}")
    print(f"{'=' * 60}")

    if result.trades:
        print(f"\n  最近 10 笔交易:")
        print(f"  {'日期':<12} {'方向':<10} {'盈亏':>10} {'原因':<15}")
        print(f"  {'-' * 50}")
        for t in result.trades[-10:]:
            print(f"  {t.entry_date:<12} {t.direction:<10} ${t.pnl:>9.2f} {t.reason:<15}")

    # Statistical validation
    if result.total_trades > 0:
        from backtest.validation import validate_backtest, print_validation_report
        sv = validate_backtest(
            strategy=result.strategy,
            sharpe=result.sharpe,
            n_trades=result.total_trades,
            win_rate=result.win_rate,
            n_trials=n_trials,
        )
        print_validation_report(sv)
