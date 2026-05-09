"""
Momentum Rotation Strategy — 月度 ETF 动量轮动.

Selects top-N ETFs by 12M-1M momentum, rebalances monthly.
Uses whole shares only (Moomoo API limitation).
Includes SMA200 trend filter: skip ETFs below their 200-day SMA.

Academic basis: Jegadeesh & Titman (1993), Moskowitz et al. (2012).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from futu import (
    OpenQuoteContext, OpenSecTradeContext, RET_OK,
    TrdEnv, TrdSide, OrderType,
)

logger = logging.getLogger(__name__)


@dataclass
class MomentumConfig:
    pool: list[str] = field(default_factory=lambda: [
        "US.SGOV", "US.BIL", "US.TLT", "US.VEA",
        "US.EEM", "US.XLF", "US.XLE", "US.IWM",
    ])
    safe_haven: str = "US.SGOV"
    budget: float = 500
    top_n: int = 2
    rebalance_day: int = 1
    lookback_months: int = 12
    skip_recent_months: int = 1
    sma_filter: int = 200
    trade_env: str = "REAL"

    @classmethod
    def from_yaml(cls, cfg: dict) -> MomentumConfig:
        m = cfg.get("momentum", {})
        default_pool = [
            "US.SGOV", "US.BIL", "US.TLT", "US.VEA",
            "US.EEM", "US.XLF", "US.XLE", "US.IWM",
        ]
        return cls(
            pool=m.get("pool", default_pool),
            budget=m.get("budget", 500),
            top_n=m.get("top_n", 2),
            rebalance_day=m.get("rebalance_day", 1),
            lookback_months=m.get("lookback_months", 12),
            skip_recent_months=m.get("skip_recent_months", 1),
            sma_filter=m.get("sma_filter", 200),
        )


@dataclass
class MomentumSignal:
    symbol: str
    momentum_score: float
    above_sma: bool
    price: float
    sma_value: float


class MomentumRotation:
    """Monthly ETF momentum rotation with SMA trend filter."""

    def __init__(self, config: MomentumConfig,
                 quote_ctx: OpenQuoteContext,
                 trade_ctx: OpenSecTradeContext,
                 notifier=None):
        self.cfg = config
        self.quote_ctx = quote_ctx
        self.trade_ctx = trade_ctx
        self.notifier = notifier
        self.trd_env = TrdEnv.REAL if config.trade_env == "REAL" else TrdEnv.SIMULATE

    # ── Signal Computation ──

    def compute_signals(self) -> list[MomentumSignal]:
        """Compute momentum scores for all pool ETFs using live data."""
        from futu import KLType, SubType

        signals = []
        for sym in self.cfg.pool:
            try:
                ret_sub, _ = self.quote_ctx.subscribe([sym], [SubType.K_DAY])
                ret, klines = self.quote_ctx.get_cur_kline(
                    sym, 300, KLType.K_DAY,
                )
                if ret != RET_OK or klines.empty or len(klines) < self.cfg.sma_filter + 1:
                    logger.warning("  %s: 数据不足 (%d bars)", sym, len(klines) if ret == RET_OK else 0)
                    continue

                closes = klines["close"].values
                current_price = closes[-1]

                # 12M-1M momentum
                lookback_bars = self.cfg.lookback_months * 21
                skip_bars = self.cfg.skip_recent_months * 21
                if len(closes) < lookback_bars + 1:
                    continue

                price_12m_ago = closes[-(lookback_bars + 1)]
                price_1m_ago = closes[-(skip_bars + 1)] if skip_bars > 0 else current_price
                mom_12m = (price_1m_ago / price_12m_ago) - 1.0

                # SMA filter
                sma = float(np.mean(closes[-self.cfg.sma_filter:]))
                above_sma = current_price > sma

                signals.append(MomentumSignal(
                    symbol=sym,
                    momentum_score=mom_12m,
                    above_sma=above_sma,
                    price=current_price,
                    sma_value=sma,
                ))

                logger.info("  %s: 动量=%.1f%% | 价格=$%.2f | SMA%d=$%.2f | %s",
                             sym, mom_12m * 100, current_price, self.cfg.sma_filter,
                             sma, "上方" if above_sma else "下方")

            except Exception as e:
                logger.warning("  %s: 计算失败: %s", sym, e)
                continue

        return signals

    def compute_signals_factor_library(self) -> list[MomentumSignal]:
        """Use pre-computed factor library data for momentum scoring.

        Loads MOM_12M_1M, MOM_6M, PRICE_SMA200, MAX_DD_60D, VOL_20D from
        the factor library and combines them using the momentum_rotation model.
        Falls back to standard compute_signals on any failure.
        """
        try:
            from factor_library.storage import load_factors
            from factor_library.search import build_factor_matrix
            from factor_library.screener import score_stocks

            categories = ["technical", "risk", "volatility", "liquidity"]
            factor_dfs = {}
            for cat in categories:
                df = load_factors(cat)
                if not df.empty:
                    factor_dfs[cat] = df

            if not factor_dfs:
                logger.info("Factor library empty, falling back to standard signals")
                return self.compute_signals()

            matrix = build_factor_matrix(factor_dfs)
            if matrix.empty:
                logger.info("Factor matrix empty, falling back to standard signals")
                return self.compute_signals()

            pool_tickers = [s.replace("US.", "") for s in self.cfg.pool]
            available = [t for t in pool_tickers if t in matrix.index]
            if len(available) < 2:
                logger.info("Too few pool ETFs in factor library (%d), "
                            "falling back to standard signals", len(available))
                return self.compute_signals()

            pool_matrix = matrix.loc[available]
            results = score_stocks(pool_matrix, model="momentum_rotation",
                                   top_n=len(available))

            score_map = dict(zip(results["symbol"], results["score"]))

            signals = []
            for sym in self.cfg.pool:
                ticker = sym.replace("US.", "")
                if ticker not in score_map:
                    continue

                # Get live price and SMA from Futu for execution
                try:
                    from futu import KLType, SubType
                    ret_sub, _ = self.quote_ctx.subscribe([sym], [SubType.K_DAY])
                    ret, klines = self.quote_ctx.get_cur_kline(
                        sym, self.cfg.sma_filter + 10, KLType.K_DAY)
                    if ret != RET_OK or klines.empty:
                        continue
                    closes = klines["close"].values
                    current_price = closes[-1]
                    sma = float(np.mean(closes[-self.cfg.sma_filter:]))
                    above_sma = current_price > sma
                except Exception:
                    # Use factor library SMA signal as fallback
                    sma200 = pool_matrix.loc[ticker].get("PRICE_SMA200", 0)
                    above_sma = sma200 > 0 if not pd.isna(sma200) else True
                    current_price = 0
                    sma = 0

                signals.append(MomentumSignal(
                    symbol=sym,
                    momentum_score=score_map[ticker],
                    above_sma=above_sma,
                    price=current_price,
                    sma_value=sma,
                ))

                mom_pct = score_map[ticker] * 100
                logger.info("  %s: 因子库动量=%.2f%% | SMA%d %s",
                            sym, mom_pct, self.cfg.sma_filter,
                            "上方" if above_sma else "下方")

            if signals:
                return signals

            logger.info("Factor library signals empty, falling back to standard")
            return self.compute_signals()

        except Exception as e:
            logger.warning("Factor library signals failed (%s), using standard", e)
            return self.compute_signals()

    def compute_signals_factor_enhanced(self) -> list[MomentumSignal]:
        """Factor-enhanced momentum: use IC-optimal lookback + turnover filter.

        Falls back to standard compute_signals on any failure.
        """
        try:
            from factor.data_provider import FactorDataProvider
            from factor.technical import calc_momentum, calc_turnover

            provider = FactorDataProvider(quote_ctx=self.quote_ctx)
            prices, volumes = provider.get_daily_panel(self.cfg.pool, years=2)
            if prices.empty or len(prices) < 252:
                logger.info("Factor data insufficient, falling back to standard signals")
                return self.compute_signals()

            mom_6m = calc_momentum(prices, 126)
            mom_12m_1m = calc_momentum(prices, 252) - calc_momentum(prices, 21)
            turnover = calc_turnover(volumes, 20)

            latest_prices = prices.iloc[-1]
            sma_values = prices.rolling(self.cfg.sma_filter).mean().iloc[-1]

            signals = []
            for sym in self.cfg.pool:
                from data.downloader import _normalize_symbol
                norm = _normalize_symbol(sym)
                if norm not in prices.columns:
                    continue

                current_price = latest_prices.get(norm, 0)
                if current_price <= 0 or pd.isna(current_price):
                    continue

                sma_val = sma_values.get(norm, 0)
                above_sma = current_price > sma_val if sma_val > 0 else True

                m6 = mom_6m[norm].iloc[-1] if norm in mom_6m.columns else 0
                m12_1 = mom_12m_1m[norm].iloc[-1] if norm in mom_12m_1m.columns else 0
                turn = turnover[norm].iloc[-1] if norm in turnover.columns else 1.0

                composite = 0.6 * (m6 if not pd.isna(m6) else 0) + \
                            0.4 * (m12_1 if not pd.isna(m12_1) else 0)

                if not pd.isna(turn) and turn > 2.0:
                    composite *= 0.7
                    logger.info("  %s: 高换手 (%.1fx) 动量打折", sym, turn)

                signals.append(MomentumSignal(
                    symbol=sym,
                    momentum_score=composite,
                    above_sma=above_sma,
                    price=current_price,
                    sma_value=sma_val if not pd.isna(sma_val) else 0,
                ))

                logger.info("  %s: 因子动量=%.1f%% (6M=%.1f%% 12M-1M=%.1f%%) | "
                            "SMA%d %s",
                            sym, composite * 100,
                            (m6 if not pd.isna(m6) else 0) * 100,
                            (m12_1 if not pd.isna(m12_1) else 0) * 100,
                            self.cfg.sma_filter,
                            "上方" if above_sma else "下方")

            return signals if signals else self.compute_signals()
        except Exception as e:
            logger.warning("Factor-enhanced signals failed (%s), using standard", e)
            return self.compute_signals()

    def select_targets(self, signals: list[MomentumSignal]) -> list[MomentumSignal]:
        """Select top-N ETFs passing the SMA filter."""
        eligible = [s for s in signals if s.above_sma]

        eligible.sort(key=lambda s: s.momentum_score, reverse=True)

        selected = eligible[:self.cfg.top_n]

        # If fewer than top_n pass filter, fill with safe haven
        if len(selected) < self.cfg.top_n:
            safe = [s for s in signals if s.symbol == self.cfg.safe_haven]
            if safe and safe[0] not in selected:
                selected.append(safe[0])

        return selected

    # ── Execution ──

    def get_current_positions(self) -> dict[str, int]:
        """Get current momentum portfolio holdings."""
        ret, data = self.trade_ctx.position_list_query(trd_env=self.trd_env)
        if ret != RET_OK or data.empty:
            return {}
        positions = {}
        for _, row in data.iterrows():
            code = str(row["code"])
            if code in self.cfg.pool:
                positions[code] = int(row["qty"])
        return positions

    def compute_target_allocation(self, targets: list[MomentumSignal]) -> dict[str, int]:
        """Compute target whole-share allocation for budget."""
        if not targets:
            return {}

        per_slot = self.cfg.budget / len(targets)
        allocation = {}
        for sig in targets:
            if sig.price <= 0:
                continue
            shares = int(per_slot / sig.price)
            if shares >= 1:
                allocation[sig.symbol] = shares
        return allocation

    def rebalance(self, dry_run: bool = False) -> dict:
        """Execute monthly rebalance.

        Returns summary of actions taken.
        """
        logger.info("动量轮动: 开始月度再平衡...")
        print("\n" + "=" * 60)
        print("  动量轮动 — 月度再平衡")
        print("=" * 60)

        # Step 1: Compute signals (factor library preferred, fallback to standard)
        signals = self.compute_signals_factor_library()
        if not signals:
            logger.warning("无有效信号")
            return {"error": "no_signals"}

        # Step 2: Select targets
        targets = self.select_targets(signals)
        target_names = [t.symbol for t in targets]
        logger.info("目标持仓: %s", target_names)
        print(f"\n  目标: {target_names}")

        # Step 3: Compute target allocation
        target_alloc = self.compute_target_allocation(targets)
        print(f"  目标配置: {target_alloc}")

        # Step 4: Get current positions
        current = self.get_current_positions()
        print(f"  当前持仓: {current}")

        # Step 5: Compute trades
        sells = []
        buys = []

        # Sell positions not in target
        for sym, qty in current.items():
            if sym not in target_alloc and qty > 0:
                sells.append({"symbol": sym, "qty": qty, "side": "SELL"})
            elif sym in target_alloc and qty > target_alloc[sym]:
                sells.append({"symbol": sym, "qty": qty - target_alloc[sym], "side": "SELL"})

        # Buy new positions or add to existing
        for sym, target_qty in target_alloc.items():
            current_qty = current.get(sym, 0)
            if current_qty < target_qty:
                buys.append({"symbol": sym, "qty": target_qty - current_qty, "side": "BUY"})

        if not sells and not buys:
            print("  无需调仓")
            return {"action": "no_change", "targets": target_names}

        # Step 6: Execute (sells first to free cash)
        results = {"sells": [], "buys": [], "targets": target_names}

        for trade in sells:
            print(f"  卖出: {trade['symbol']} x {trade['qty']}")
            if not dry_run:
                r = self._execute_order(trade["symbol"], trade["qty"], TrdSide.SELL)
                results["sells"].append({**trade, "success": r is not None})

        for trade in buys:
            print(f"  买入: {trade['symbol']} x {trade['qty']}")
            if not dry_run:
                r = self._execute_order(trade["symbol"], trade["qty"], TrdSide.BUY)
                results["buys"].append({**trade, "success": r is not None})

        # Step 7: Notify
        self._notify_rebalance(signals, targets, results, dry_run)
        return results

    def _execute_order(self, symbol: str, qty: int, side: TrdSide) -> Optional[str]:
        price = self._get_price(symbol)
        if price <= 0:
            logger.error("无法获取 %s 报价", symbol)
            return None

        ret, data = self.trade_ctx.place_order(
            price=round(price, 2),
            qty=qty,
            code=symbol,
            trd_side=side,
            order_type=OrderType.NORMAL,
            trd_env=self.trd_env,
        )
        if ret == RET_OK:
            order_id = str(data["order_id"].iloc[0])
            logger.info("  下单成功: %s %s x %d @ $%.2f | ID=%s",
                         "买入" if side == TrdSide.BUY else "卖出",
                         symbol, qty, price, order_id)
            return order_id
        else:
            logger.error("  下单失败: %s %s x %d: %s",
                         "买入" if side == TrdSide.BUY else "卖出",
                         symbol, qty, data)
            return None

    def _get_price(self, symbol: str) -> float:
        ret, snap = self.quote_ctx.get_market_snapshot([symbol])
        if ret != RET_OK or snap.empty:
            return 0.0
        return float(snap["last_price"].iloc[0])

    def _notify_rebalance(self, signals, targets, results, dry_run):
        mode = "[DRY RUN] " if dry_run else ""
        lines = [f"<b>{mode}动量轮动 月度再平衡</b>\n"]

        lines.append("📊 动量排名:")
        for s in sorted(signals, key=lambda x: x.momentum_score, reverse=True):
            mark = " ✓" if s in targets else ""
            sma_mark = "↑" if s.above_sma else "↓"
            lines.append(f"  {s.symbol}: {s.momentum_score:+.1%} {sma_mark}{mark}")

        if results.get("sells"):
            lines.append("\n卖出:")
            for t in results["sells"]:
                ok = "✓" if t.get("success") else "✗"
                lines.append(f"  {ok} {t['symbol']} x {t['qty']}")

        if results.get("buys"):
            lines.append("\n买入:")
            for t in results["buys"]:
                ok = "✓" if t.get("success") else "✗"
                lines.append(f"  {ok} {t['symbol']} x {t['qty']}")

        if not results.get("sells") and not results.get("buys"):
            lines.append("\n无需调仓")

        msg = "\n".join(lines)
        if self.notifier:
            try:
                self.notifier.send_message(msg)
            except Exception as e:
                logger.warning("Telegram 通知失败: %s", e)

    # ── Backtest Support ──

    @staticmethod
    def backtest_momentum(daily_data: dict[str, pd.DataFrame],
                          budget: float = 500,
                          top_n: int = 2,
                          lookback: int = 252,
                          skip: int = 21,
                          sma_period: int = 200,
                          safe_haven: str = "SGOV") -> dict:
        """Pure backtest: monthly rebalance on historical data.

        Args:
            daily_data: {symbol: DataFrame with 'time_key', 'close'}

        Returns dict with equity curve, trades, metrics.
        """
        all_dates = set()
        for df in daily_data.values():
            all_dates.update(df["time_key"].tolist())
        dates = sorted(all_dates)

        # Build price matrix
        price_df = pd.DataFrame(index=dates)
        for sym, df in daily_data.items():
            temp = df.set_index("time_key")["close"]
            price_df[sym] = temp
        price_df = price_df.sort_index().ffill()

        equity = [budget]
        trades_log = []
        holdings: dict[str, int] = {}
        cash = budget

        rebalance_months = set()

        for i in range(lookback + 1, len(price_df)):
            date = price_df.index[i]
            month_key = str(date)[:7]

            # Monthly rebalance on first trading day of month
            if month_key in rebalance_months:
                # Just update equity
                port_value = cash
                for sym, qty in holdings.items():
                    if sym in price_df.columns:
                        port_value += qty * price_df[sym].iloc[i]
                equity.append(port_value)
                continue

            rebalance_months.add(month_key)

            # Compute momentum for all symbols
            scores = {}
            for sym in daily_data.keys():
                if sym not in price_df.columns:
                    continue
                closes = price_df[sym].iloc[:i + 1].values

                if len(closes) < lookback + 1:
                    continue

                p_end = closes[-skip - 1] if skip > 0 else closes[-1]
                p_start = closes[-lookback - 1]
                if p_start <= 0 or np.isnan(p_start) or np.isnan(p_end):
                    continue
                mom = (p_end / p_start) - 1.0

                current = closes[-1]
                if np.isnan(current) or current <= 0:
                    continue

                sma = float(np.nanmean(closes[-sma_period:])) if len(closes) >= sma_period else 0
                above = current > sma if sma > 0 else True

                scores[sym] = {"mom": mom, "above_sma": above, "price": current}

            # Select top N passing SMA filter
            eligible = {s: v for s, v in scores.items() if v["above_sma"]}
            ranked = sorted(eligible.items(), key=lambda x: x[1]["mom"], reverse=True)
            selected = [s for s, _ in ranked[:top_n]]

            if len(selected) < top_n and safe_haven in scores:
                if safe_haven not in selected:
                    selected.append(safe_haven)

            # Sell all current (with cost)
            COST_BPS = 8  # 5 slippage + 3 spread
            COMMISSION = 1.0
            for sym, qty in holdings.items():
                if qty > 0 and sym in price_df.columns:
                    sell_price = price_df[sym].iloc[i]
                    notional = qty * sell_price
                    tc = COMMISSION + notional * COST_BPS / 10000
                    cash += notional - tc
            holdings = {}

            # Buy new (with cost)
            if selected:
                per_slot = cash / len(selected)
                for sym in selected:
                    price = scores[sym]["price"]
                    if price <= 0:
                        continue
                    qty = int(per_slot / price)
                    if qty >= 1:
                        notional = qty * price
                        tc = COMMISSION + notional * COST_BPS / 10000
                        cash -= notional + tc
                        holdings[sym] = qty
                        trades_log.append({
                            "date": str(date)[:10], "symbol": sym,
                            "qty": qty, "price": price, "action": "BUY",
                        })

            port_value = cash
            for sym, qty in holdings.items():
                if sym in price_df.columns:
                    port_value += qty * price_df[sym].iloc[i]
            equity.append(port_value)

        equity = np.array(equity)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]

        total_return = (equity[-1] / equity[0]) - 1.0 if len(equity) > 1 else 0
        n_years = len(price_df) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(np.min(dd))

        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

        return {
            "equity": equity,
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "n_trades": len(trades_log),
            "trades": trades_log,
            "final_value": equity[-1],
        }

    def is_rebalance_day(self) -> bool:
        """Check if today is a rebalance day (first trading day of month)."""
        today = datetime.now()
        if today.day > 5:
            return False
        if today.weekday() >= 5:
            return False
        yesterday = today - timedelta(days=1)
        return yesterday.month != today.month or today.day == 1

    def status(self) -> dict:
        """Return current momentum portfolio status."""
        positions = self.get_current_positions()
        total_value = 0
        details = []
        for sym, qty in positions.items():
            price = self._get_price(sym)
            value = qty * price
            total_value += value
            details.append({"symbol": sym, "qty": qty, "price": price, "value": value})
        return {
            "total_value": total_value,
            "budget": self.cfg.budget,
            "positions": details,
            "is_rebalance_day": self.is_rebalance_day(),
        }
