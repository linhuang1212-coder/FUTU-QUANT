"""FUTU-QUANT Multi-Strategy Live Trading Runner

Swing-only system with macro risk management:
  - QQQ SMA200 trend filter (global entry gate)
  - VIX adaptive continuous position sizing
  - Dual momentum monthly rotation (TQQQ vs SOXL)
  - Multiple swing strategies per symbol

Usage:
    python run_live.py              # REAL trading
    python run_live.py --dry-run    # Test without placing orders
    python run_live.py --once       # Evaluate once and exit (for cron/scheduler)

IMPORTANT: Ensure FutuOpenD is running and logged in before starting.
"""

import sys
import io
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.breakout import BreakoutStrategy
from strategy.rsi_reversal import RsiReversalStrategy
from strategy.multi_factor import MultiFactorStrategy
from strategy.base import SignalDirection, Signal
from data.indicators import TechnicalIndicators
from risk.pdt_guard import PdtGuard
from risk.vol_target import VolatilityTargetManager, MarketRegime
from utils.logger import setup_logger
from utils.helpers import load_yaml, get_project_root

STRATEGY_CLASSES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "rsi_reversal": RsiReversalStrategy,
    "multi_factor": MultiFactorStrategy,
}

TREND_STRATEGIES = {"momentum", "breakout"}
REVERSION_STRATEGIES = {"mean_reversion", "rsi_reversal"}


class MultiStrategyTrader:
    def __init__(self, config_path: str, dry_run: bool = False):
        self.root = get_project_root()
        self.config = load_yaml(str(self.root / config_path))
        self.dry_run = dry_run

        log_name = "live-dry" if dry_run else "live"
        self.logger = setup_logger(
            log_name, str(self.root / "data_store" / "logs" / f"{log_name}.log")
        )

        self.trade_log_dir = self.root / "data_store" / "trades"
        self.trade_log_dir.mkdir(parents=True, exist_ok=True)

        self.slots = self._load_slots()

        pdt_cfg = self.config.get("pdt", {})
        self.pdt = PdtGuard(
            max_day_trades=pdt_cfg.get("max_day_trades", 3),
            rolling_window_days=pdt_cfg.get("rolling_window_days", 5),
        )
        self.pdt_enabled = pdt_cfg.get("enabled", True)

        self._quote_ctx = None
        self._trade_ctx = None

        # Volatility target / regime manager
        vtm_cfg = self.config.get("vol_target", {})
        self.vtm = VolatilityTargetManager(
            vix_entry_max=vtm_cfg.get("vix_entry_max", 28.0),
            vix_force_close=vtm_cfg.get("vix_force_close", 35.0),
            vix_reduce_threshold=vtm_cfg.get("vix_reduce_threshold", 22.0),
            adx_trend_threshold=vtm_cfg.get("adx_trend_threshold", 25.0),
            adx_period=vtm_cfg.get("adx_period", 14),
            vol_target=vtm_cfg.get("vol_target", 0.18),
            ewma_lambda=vtm_cfg.get("ewma_lambda", 0.94),
            dd_threshold=vtm_cfg.get("dd_threshold", -0.20),
            dd_scale_factor=vtm_cfg.get("dd_scale_factor", 0.5),
            max_position_scale=vtm_cfg.get("max_position_scale", 0.95),
        )
        self._regime: Optional[MarketRegime] = None

        # Swing position (held overnight)
        self.holding_symbol: Optional[str] = None
        self.holding_qty: int = 0
        self.holding_avg_price: float = 0.0
        self.holding_strategy: str = ""

        # Intraday position (must close before EOD)
        self.intraday_symbol: Optional[str] = None
        self.intraday_qty: int = 0
        self.intraday_avg_price: float = 0.0
        self.intraday_strategy: str = ""

    # ── Config loading ──────────────────────────────────────────

    def _load_slots(self) -> list[dict]:
        portfolio = self.config.get("portfolio", {})
        symbol_cfgs = portfolio.get("symbols", [])
        slots = []
        for sym_cfg in symbol_cfgs:
            code = sym_cfg["code"]
            strats = []
            for s in sym_cfg.get("strategies", []):
                name = s["name"]
                cls = STRATEGY_CLASSES.get(name)
                if cls is None:
                    self.logger.warning(f"Unknown strategy '{name}', skipping")
                    continue
                strats.append({
                    "name": name,
                    "strategy": cls(params=s.get("params", {})),
                    "sharpe_weight": s.get("sharpe_weight", 1.0),
                })
            if strats:
                slots.append({"code": code, "strategies": strats})
                self.logger.info(
                    f"Loaded {code}: {', '.join(s['name'] for s in strats)}"
                )
        return slots

    def _all_symbols(self) -> list[str]:
        return [s["code"] for s in self.slots]

    # ── Connection ──────────────────────────────────────────────

    def connect(self) -> bool:
        from futu import OpenQuoteContext, OpenSecTradeContext

        try:
            self._quote_ctx = OpenQuoteContext(
                host=self.config["futu"]["host"],
                port=self.config["futu"]["port"],
            )
            self.logger.info("Quote context connected")
        except Exception as e:
            self.logger.error(f"Quote connection failed: {e}")
            return False

        if not self.dry_run:
            try:
                self._trade_ctx = OpenSecTradeContext(
                    host=self.config["futu"]["host"],
                    port=self.config["futu"]["port"],
                )
                env_str = self.config["futu"]["trade_env"]
                self.logger.info(f"Trade context connected ({env_str})")
            except Exception as e:
                self.logger.error(f"Trade connection failed: {e}")
                return False

        return True

    def disconnect(self):
        if self._quote_ctx:
            self._quote_ctx.close()
        if self._trade_ctx:
            self._trade_ctx.close()
        self.logger.info("Disconnected")

    # ── Position sync ───────────────────────────────────────────

    def sync_position(self):
        """Sync swing position from broker. Intraday position is session-local."""
        self.holding_symbol = None
        self.holding_qty = 0
        self.holding_avg_price = 0.0
        self.holding_strategy = ""

        if self._trade_ctx is None:
            return

        from futu import RET_OK, TrdEnv
        env = TrdEnv.REAL if self.config["futu"]["trade_env"] == "REAL" else TrdEnv.SIMULATE
        ret, data = self._trade_ctx.position_list_query(trd_env=env)
        if ret != RET_OK or data is None:
            return

        tracked = set(self._all_symbols())
        for _, row in data.iterrows():
            code = row["code"]
            qty = int(row["qty"])
            if code in tracked and qty > 0:
                self.holding_symbol = code
                self.holding_qty = qty
                self.holding_avg_price = float(row["cost_price"])
                self.logger.info(
                    f"Existing position: {code} x{qty} @ ${self.holding_avg_price:.2f}"
                )
                return

        self.logger.info("No existing position in tracked symbols")

    # ── Data fetching ───────────────────────────────────────────

    def get_daily_kline(self, symbol: str, count: int = 60) -> Optional[pd.DataFrame]:
        from futu import RET_OK, KLType

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=count * 2)).strftime("%Y-%m-%d")

        ret, data, _ = self._quote_ctx.request_history_kline(
            symbol, start=start_date, end=end_date,
            ktype=KLType.K_DAY, max_count=count,
        )
        if ret == RET_OK and data is not None and len(data) >= 20:
            return data
        self.logger.error(f"Failed to get daily kline for {symbol}")
        return None

    def get_5min_kline(self, symbol: str, count: int = 60) -> Optional[pd.DataFrame]:
        from futu import RET_OK, KLType

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        ret, data, _ = self._quote_ctx.request_history_kline(
            symbol, start=start_date, end=end_date,
            ktype=KLType.K_5M, max_count=count,
        )
        if ret == RET_OK and data is not None and len(data) >= 20:
            return data
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        from futu import RET_OK
        ret, data = self._quote_ctx.get_market_snapshot([symbol])
        if ret == RET_OK and data is not None and len(data) > 0:
            return float(data.iloc[0]["last_price"])
        return None

    # ── Indicator precomputation ────────────────────────────────

    @staticmethod
    def _precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for p in (5, 8, 10, 14, 15, 20):
            out = TechnicalIndicators.add_ma(out, p)
            out = TechnicalIndicators.add_ema(out, p)
        for p in (5, 7, 10, 14):
            out = TechnicalIndicators.add_rsi(out, p)
        for bp, bs in ((15, 2.0), (20, 2.0)):
            out = TechnicalIndicators.add_bollinger(out, bp, bs)
        out = TechnicalIndicators.add_atr(out, 14)
        out = TechnicalIndicators.add_macd(out, 12, 26, 9)
        return out

    @staticmethod
    def _precompute_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = TechnicalIndicators.add_rsi(out, 14)
        out = TechnicalIndicators.add_ema(out, 8)
        out = TechnicalIndicators.add_ema(out, 20)
        out = TechnicalIndicators.add_atr(out, 14)
        tp = (out["high"] + out["low"] + out["close"]) / 3
        cum_tp_vol = (tp * out["volume"]).cumsum()
        cum_vol = out["volume"].cumsum().replace(0, float("nan"))
        out["vwap"] = cum_tp_vol / cum_vol
        out["vol_ma20"] = out["volume"].rolling(20).mean()
        return out

    # ── Regime assessment ──────────────────────────────────────

    def _assess_market_regime(self) -> MarketRegime:
        """Fetch VIX, compute ADX/EWMA vol/drawdown, return MarketRegime."""
        vix = self.vtm.get_vix_level(self._quote_ctx) if self._quote_ctx else None

        ref_symbol = self._all_symbols()[0] if self._all_symbols() else "US.TQQQ"
        df = self.get_daily_kline(ref_symbol, 60)
        if df is not None and len(df) >= 30:
            adx = self.vtm.compute_adx(df)
            ewma_vol = self.vtm.compute_ewma_vol(df)
            drawdown = self.vtm.compute_drawdown(df)
        else:
            adx, ewma_vol, drawdown = 0.0, 0.3, 0.0

        regime = self.vtm.assess_regime(vix, adx, ewma_vol, drawdown)
        self._regime = regime
        self.logger.info(
            f"[REGIME] {regime.regime_label} | VIX={regime.vix_level:.1f} "
            f"ADX={regime.adx_value:.1f} trending={regime.is_trending} "
            f"scale={regime.position_scale:.2f} "
            f"vix_ok={regime.vix_ok} danger={regime.vix_danger}"
        )
        return regime

    # ── QQQ SMA200 Trend Filter ──────────────────────────────

    def _check_qqq_sma200(self) -> bool:
        """Global entry gate: QQQ must be above its 200-day SMA.
        Validated on 10yr real data — avoids bear markets entirely."""
        df = self.get_daily_kline("US.QQQ", 250)
        if df is None or len(df) < 200:
            self.logger.warning("[SMA200] Cannot get QQQ data, defaulting to ALLOW")
            return True

        sma200 = df["close"].rolling(200).mean().iloc[-1]
        qqq_close = df["close"].iloc[-1]
        above = qqq_close > sma200

        self.logger.info(
            f"[SMA200] QQQ={qqq_close:.2f} vs SMA200={sma200:.2f} "
            f"-> {'ABOVE (trade OK)' if above else 'BELOW (no new entries)'}"
        )
        return above

    # ── Dual Momentum Rotation ───────────────────────────────

    def _dual_momentum_rank(self) -> Optional[str]:
        """Monthly rotation: rank TQQQ vs SOXL by 1m+3m momentum.
        Returns preferred symbol or None if both negative."""
        symbols = {"US.TQQQ": None, "US.SOXL": None}

        for sym in symbols:
            df = self.get_daily_kline(sym, 80)
            if df is None or len(df) < 63:
                continue
            time.sleep(0.3)
            close = df["close"].values
            mom_1m = close[-1] / close[-21] - 1 if len(close) >= 21 else 0
            mom_3m = close[-1] / close[-63] - 1 if len(close) >= 63 else 0
            symbols[sym] = 0.5 * mom_1m + 0.5 * mom_3m

        scores = {k: v for k, v in symbols.items() if v is not None}
        if not scores:
            return None

        best_sym = max(scores, key=scores.get)
        best_score = scores[best_sym]

        for sym, score in scores.items():
            self.logger.info(
                f"[MOMENTUM] {sym}: score={score:+.3f} "
                f"({'selected' if sym == best_sym and best_score > 0 else 'skip'})"
            )

        if best_score <= 0:
            self.logger.info("[MOMENTUM] All negative -> stay cash")
            return None
        return best_sym

    # ── VIX Adaptive Position Sizing (enhanced) ──────────────

    def _vix_adaptive_allocation(self, base_alloc: float = 0.95) -> float:
        """Continuous VIX-based position sizing (enhanced from binary on/off).
        Returns allocation fraction 0.0 - 0.95."""
        if self._regime is None:
            return base_alloc

        vix = self._regime.vix_level
        if vix < 15:
            alloc = 0.95
        elif vix < 20:
            alloc = 0.75
        elif vix < 28:
            alloc = 0.50
        else:
            alloc = 0.0

        vol_scale = self._regime.position_scale
        final = min(alloc, base_alloc) * vol_scale

        if final < base_alloc:
            self.logger.info(
                f"[VIX SIZING] VIX={vix:.1f} -> base={alloc:.0%}, "
                f"vol_scale={vol_scale:.2f}, final={final:.1%}"
            )
        return final

    # ── Swing layer (daily) ─────────────────────────────────────

    def _collect_swing_signals(self) -> list[dict]:
        regime = self._regime
        results = []
        for slot in self.slots:
            code = slot["code"]
            df = self.get_daily_kline(code, 60)
            if df is None:
                continue
            time.sleep(0.3)
            df = self._precompute_indicators(df)
            for strat_cfg in slot["strategies"]:
                sname = strat_cfg["name"]

                try:
                    signal = strat_cfg["strategy"].on_bar(code, df)
                except Exception as e:
                    self.logger.error(f"Strategy {sname}@{code} error: {e}")
                    continue
                if signal is None:
                    continue

                # ADX filter: block trend-strategy BUY when market is not trending
                if (regime and signal.direction == SignalDirection.BUY
                        and sname in TREND_STRATEGIES
                        and not regime.is_trending):
                    self.logger.info(
                        f"  [SWING {sname}@{code}] BUY blocked by ADX filter "
                        f"(ADX={regime.adx_value:.1f}<{self.vtm.adx_trend_threshold})"
                    )
                    continue

                score = strat_cfg["sharpe_weight"] * signal.strength
                results.append({
                    "score": score,
                    "symbol": code,
                    "signal": signal,
                    "strategy_name": sname,
                    "sharpe_weight": strat_cfg["sharpe_weight"],
                    "layer": "swing",
                })
                self.logger.info(
                    f"  [SWING {sname}@{code}] "
                    f"{signal.direction.value} str={signal.strength:.1f} "
                    f"score={score:.1f} | {signal.reason}"
                )
        return results

    # ── Intraday layer v2 (trend-following, 5-min) ──────────────

    INTRADAY_CONFIGS = {}  # cleared: all intraday strategies failed on real 5min data

    def _get_today_5min(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get today's 5-min bars only (for day-boundary-aware strategies)."""
        df5 = self.get_5min_kline(symbol, 80)
        if df5 is None or len(df5) < 10:
            return None
        df5 = self._precompute_intraday_indicators(df5)
        df5["date"] = pd.to_datetime(df5["time_key"]).dt.date
        today = df5["date"].iloc[-1]
        today_df = df5[df5["date"] == today].reset_index(drop=True)
        return today_df if len(today_df) >= 6 else None

    _INTRADAY_EVAL = {}

    def _evaluate_intraday_entry(self) -> Optional[dict]:
        """Scan symbols with trend-following intraday strategies."""
        if self.holding_symbol is not None or self.intraday_symbol is not None:
            return None

        best = None
        for sym, strats in self.INTRADAY_CONFIGS.items():
            today_df = self._get_today_5min(sym)
            if today_df is None:
                continue
            time.sleep(0.3)

            for strat_cfg in strats:
                fn_name = self._INTRADAY_EVAL.get(strat_cfg["name"])
                if fn_name is None:
                    continue
                fn = getattr(self, fn_name)
                result = fn(today_df, strat_cfg["params"])
                if result is None:
                    continue

                entry = {
                    "score": result["strength"],
                    "symbol": sym,
                    "strength": result["strength"],
                    "reason": result["reason"],
                    "layer": "intraday",
                    "strategy_name": strat_cfg["name"],
                    "stop": result.get("stop"),
                    "target": result.get("target"),
                }
                if best is None or entry["score"] > best["score"]:
                    best = entry

        return best

    def _evaluate_intraday_exit(self) -> Optional[str]:
        """Check if intraday position should be closed (VWAP break or stop)."""
        if self.intraday_symbol is None:
            return None

        price = self.get_current_price(self.intraday_symbol)
        if price is None:
            return None

        pnl_pct = (price / self.intraday_avg_price - 1) * 100 if self.intraday_avg_price > 0 else 0

        today_df = self._get_today_5min(self.intraday_symbol)
        if today_df is not None and "vwap" in today_df.columns:
            vwap = today_df["vwap"].iloc[-1]
            if not pd.isna(vwap) and price < vwap and pnl_pct < 0:
                return f"Price below VWAP ({pnl_pct:+.1f}%)"

        if pnl_pct <= -3.0:
            return f"Intraday hard stop ({pnl_pct:+.1f}%)"

        if pnl_pct >= 4.0:
            return f"Target profit reached ({pnl_pct:+.1f}%)"

        return None

    # ── Order execution ─────────────────────────────────────────

    def execute_buy(self, symbol: str, price: float, strategy_name: str,
                    is_intraday: bool = False) -> bool:
        capital = self.config["account"]["initial_capital"]
        alloc = self._vix_adaptive_allocation(0.95)
        qty = int(capital * alloc / price)

        if qty <= 0:
            self.logger.warning("Calculated quantity is 0, skipping")
            return False

        tag = "INTRADAY-BUY" if is_intraday else "BUY"
        cost = qty * price
        self.logger.info(f"[{tag}] {symbol} x{qty} @ ${price:.2f} = ${cost:.2f} ({strategy_name})")

        if self.dry_run:
            self.logger.info("[DRY-RUN] Order not placed")
            if is_intraday:
                self.intraday_symbol = symbol
                self.intraday_qty = qty
                self.intraday_avg_price = price
                self.intraday_strategy = strategy_name
            else:
                self.holding_symbol = symbol
                self.holding_qty = qty
                self.holding_avg_price = price
                self.holding_strategy = strategy_name
            self._log_trade(tag, symbol, qty, price, strategy=strategy_name, dry_run=True)
            return True

        from futu import RET_OK, TrdSide, OrderType as FutuOrderType, TrdEnv
        env = TrdEnv.REAL if self.config["futu"]["trade_env"] == "REAL" else TrdEnv.SIMULATE
        ret, data = self._trade_ctx.place_order(
            price=price, qty=qty, code=symbol,
            trd_side=TrdSide.BUY,
            order_type=FutuOrderType.MARKET,
            trd_env=env,
        )
        if ret == RET_OK:
            order_id = data.iloc[0]["order_id"] if len(data) > 0 else "N/A"
            self.logger.info(f"[{tag} CONFIRMED] order_id={order_id}")
            if is_intraday:
                self.intraday_symbol = symbol
                self.intraday_qty = qty
                self.intraday_avg_price = price
                self.intraday_strategy = strategy_name
            else:
                self.holding_symbol = symbol
                self.holding_qty = qty
                self.holding_avg_price = price
                self.holding_strategy = strategy_name
            self._log_trade(tag, symbol, qty, price, strategy=strategy_name)
            return True
        else:
            self.logger.error(f"[{tag} FAILED] {data}")
            self._log_trade(f"{tag}_FAILED", symbol, qty, price,
                            strategy=strategy_name, error=str(data))
            return False

    def execute_sell(self, price: float, is_intraday: bool = False) -> bool:
        if is_intraday:
            symbol = self.intraday_symbol
            qty = self.intraday_qty
            avg_price = self.intraday_avg_price
            strategy = self.intraday_strategy
        else:
            symbol = self.holding_symbol
            qty = self.holding_qty
            avg_price = self.holding_avg_price
            strategy = self.holding_strategy

        if qty <= 0 or symbol is None:
            self.logger.warning("No position to sell")
            return False

        pnl = (price - avg_price) * qty
        pnl_pct = (price / avg_price - 1) * 100 if avg_price > 0 else 0

        tag = "INTRADAY-SELL" if is_intraday else "SELL"
        self.logger.info(
            f"[{tag}] {symbol} x{qty} @ ${price:.2f} "
            f"PnL=${pnl:+.2f} ({pnl_pct:+.2f}%) ({strategy})"
        )

        if self.dry_run:
            self.logger.info("[DRY-RUN] Order not placed")
            self._log_trade(tag, symbol, qty, price, pnl=pnl,
                            strategy=strategy, dry_run=True)
            if is_intraday:
                self._clear_intraday()
                self.pdt.record_day_trade(symbol)
            else:
                self._clear_holding()
            return True

        from futu import RET_OK, TrdSide, OrderType as FutuOrderType, TrdEnv
        env = TrdEnv.REAL if self.config["futu"]["trade_env"] == "REAL" else TrdEnv.SIMULATE
        ret, data = self._trade_ctx.place_order(
            price=price, qty=qty, code=symbol,
            trd_side=TrdSide.SELL,
            order_type=FutuOrderType.MARKET,
            trd_env=env,
        )
        if ret == RET_OK:
            order_id = data.iloc[0]["order_id"] if len(data) > 0 else "N/A"
            self.logger.info(f"[{tag} CONFIRMED] order_id={order_id}")
            self._log_trade(tag, symbol, qty, price, pnl=pnl, strategy=strategy)
            if is_intraday:
                self._clear_intraday()
                self.pdt.record_day_trade(symbol)
            else:
                self._clear_holding()
            return True
        else:
            self.logger.error(f"[{tag} FAILED] {data}")
            self._log_trade(f"{tag}_FAILED", symbol, qty, price,
                            strategy=strategy, error=str(data))
            return False

    def _clear_holding(self):
        self.holding_symbol = None
        self.holding_qty = 0
        self.holding_avg_price = 0.0
        self.holding_strategy = ""

    def _clear_intraday(self):
        self.intraday_symbol = None
        self.intraday_qty = 0
        self.intraday_avg_price = 0.0
        self.intraday_strategy = ""

    def _log_trade(self, action: str, symbol: str, qty: int, price: float,
                   pnl: float = 0, strategy: str = "", dry_run: bool = False,
                   error: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "symbol": symbol,
            "strategy": strategy,
            "qty": qty,
            "price": price,
            "pnl": round(pnl, 2),
            "pdt_remaining": self.pdt.remaining_day_trades(),
            "dry_run": dry_run,
            "error": error,
        }
        log_file = self.trade_log_dir / f"trades_{datetime.now().strftime('%Y%m')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Can we trade intraday right now? ────────────────────────

    def _can_intraday(self) -> bool:
        if not self.pdt_enabled:
            return True
        if not self.pdt.can_day_trade():
            return False
        # Don't do intraday if we already have a swing position
        if self.holding_symbol is not None:
            return False
        return True

    def _is_market_hours(self) -> bool:
        """Rough check: 9:35 - 15:45 ET (avoids first 5 min and last 15 min)."""
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("US/Eastern"))
        t = et.time()
        from datetime import time as dtime
        return dtime(9, 35) <= t <= dtime(15, 45)

    def _is_eod_close_window(self) -> bool:
        """Last 15 minutes before market close."""
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("US/Eastern"))
        t = et.time()
        from datetime import time as dtime
        minutes_before = self.config["risk"].get("eod_close_minutes_before", 15)
        close_h, close_m = 16, 0
        close_minutes = close_h * 60 + close_m
        current_minutes = t.hour * 60 + t.minute
        return 0 <= (close_minutes - current_minutes) <= minutes_before

    # ── Main evaluation loops ───────────────────────────────────

    def run_once(self):
        """Daily swing evaluation with macro risk layers:
        1. Regime assessment (VIX/ADX/EWMA vol)
        2. QQQ SMA200 trend filter (global entry gate)
        3. Dual momentum rotation (prefer TQQQ or SOXL)
        4. Collect swing strategy signals
        5. Execute best signal with VIX adaptive position sizing
        """
        self.sync_position()
        self.pdt.cleanup_old_trades()

        self.logger.info("=" * 60)
        self.logger.info(f"[SWING] evaluation at {datetime.now().isoformat()}")
        self.logger.info(
            f"Swing: {self.holding_symbol or 'FLAT'}"
            + (f" x{self.holding_qty} @ ${self.holding_avg_price:.2f}" if self.holding_symbol else "")
        )

        regime = self._assess_market_regime()

        # VIX danger: force-close all positions immediately
        if regime.vix_danger and self.holding_symbol:
            self.logger.warning(
                f"[VIX DANGER] VIX={regime.vix_level:.1f} >= {self.vtm.vix_force_close}. "
                f"Force-closing swing position {self.holding_symbol}!"
            )
            price = self.get_current_price(self.holding_symbol)
            if price:
                self.execute_sell(price, is_intraday=False)
            return

        # SMA200 check: if QQQ below SMA200 and we hold, force exit
        qqq_above_sma200 = self._check_qqq_sma200()
        if not qqq_above_sma200 and self.holding_symbol:
            self.logger.warning(
                f"[SMA200 EXIT] QQQ below SMA200 -> closing {self.holding_symbol}"
            )
            price = self.get_current_price(self.holding_symbol)
            if price:
                self.execute_sell(price, is_intraday=False)
            return

        all_signals = self._collect_swing_signals()

        if not all_signals:
            self.logger.info("No swing signals. Done.")
            return

        sell_signals = [s for s in all_signals if s["signal"].direction == SignalDirection.SELL]
        buy_signals = sorted(
            [s for s in all_signals if s["signal"].direction == SignalDirection.BUY],
            key=lambda x: x["score"], reverse=True,
        )

        self.logger.info(f"Swing signals: {len(buy_signals)} BUY, {len(sell_signals)} SELL")

        if self.holding_symbol:
            exit_signals = [s for s in sell_signals if s["symbol"] == self.holding_symbol]
            if exit_signals:
                best_exit = max(exit_signals, key=lambda x: x["score"])
                self.logger.info(
                    f">>> SWING EXIT for {self.holding_symbol}: "
                    f"{best_exit['strategy_name']} (score={best_exit['score']:.1f}) "
                    f"| {best_exit['signal'].reason}"
                )
                price = self.get_current_price(self.holding_symbol)
                if price:
                    self.execute_sell(price, is_intraday=False)
                else:
                    self.logger.error(f"Cannot get price for {self.holding_symbol}")
                    return
            else:
                self.logger.info(
                    f"Holding {self.holding_symbol} x{self.holding_qty}, no exit signal."
                )
                return

        if buy_signals and self.holding_symbol is None:
            # Gate 1: SMA200 filter
            if not qqq_above_sma200:
                self.logger.info("[SMA200 BLOCK] QQQ below SMA200, no new entries")
                return

            # Gate 2: VIX filter
            if not regime.vix_ok:
                self.logger.info(
                    f"[VIX BLOCK] VIX={regime.vix_level:.1f} >= {self.vtm.vix_entry_max}. "
                    f"No new swing entries allowed."
                )
                return

            # Gate 3: Dual momentum — prefer the momentum-ranked symbol
            preferred_sym = self._dual_momentum_rank()
            if preferred_sym:
                preferred_buys = [b for b in buy_signals if b["symbol"] == preferred_sym]
                if preferred_buys:
                    buy_signals = preferred_buys
                    self.logger.info(
                        f"[MOMENTUM FILTER] Preferring {preferred_sym} "
                        f"({len(preferred_buys)} signals)"
                    )

            best = buy_signals[0]
            self.logger.info(
                f">>> SWING BUY: {best['strategy_name']}@{best['symbol']} "
                f"score={best['score']:.1f} str={best['signal'].strength:.1f} "
                f"| {best['signal'].reason}"
            )
            price = self.get_current_price(best["symbol"])
            if price:
                self.execute_buy(best["symbol"], price, best["strategy_name"],
                                 is_intraday=False)
            else:
                self.logger.error(f"Cannot get price for {best['symbol']}")
        elif self.holding_symbol is None:
            self.logger.info("No swing BUY signals. Staying flat.")

    def run_intraday_tick(self):
        """Single intraday evaluation tick (called every 5 min during market hours)."""
        # VIX danger: force-close intraday position immediately
        if self.intraday_symbol and self._regime and self._regime.vix_danger:
            price = self.get_current_price(self.intraday_symbol)
            if price:
                self.logger.warning(
                    f"[VIX DANGER] Force closing intraday {self.intraday_symbol}"
                )
                self.execute_sell(price, is_intraday=True)
            return

        # EOD force close
        if self.intraday_symbol and self._is_eod_close_window():
            price = self.get_current_price(self.intraday_symbol)
            if price:
                self.logger.warning(
                    f"[EOD CLOSE] Force closing intraday {self.intraday_symbol} "
                    f"before market close"
                )
                self.execute_sell(price, is_intraday=True)
            return

        # Check intraday exit
        if self.intraday_symbol:
            exit_reason = self._evaluate_intraday_exit()
            if exit_reason:
                price = self.get_current_price(self.intraday_symbol)
                if price:
                    self.logger.info(f"[INTRADAY EXIT] {exit_reason}")
                    self.execute_sell(price, is_intraday=True)
            return

        # Check intraday entry
        if not self._can_intraday():
            return

        # VIX filter: block new intraday entries
        if self._regime and not self._regime.vix_ok:
            return

        entry = self._evaluate_intraday_entry()
        if entry:
            self.logger.info(
                f"[INTRADAY ENTRY] {entry['symbol']} "
                f"str={entry['strength']:.1f} | {entry['reason']} "
                f"(PDT remaining: {self.pdt.remaining_day_trades()})"
            )
            if self.pdt.should_warn():
                self.logger.warning("[PDT WARNING] This is your LAST day trade in the window!")

            price = self.get_current_price(entry["symbol"])
            if price:
                self.execute_buy(
                    entry["symbol"], price,
                    f"intraday_{entry['reason'][:30]}",
                    is_intraday=True,
                )

    def run_loop(self, interval: int = 300):
        """Main loop: daily swing eval + intraday 5-min scanning."""
        self.sync_position()
        self.pdt.cleanup_old_trades()

        sym_list = ", ".join(self._all_symbols())
        n_strats = sum(len(s["strategies"]) for s in self.slots)
        print(f"\n{'=' * 60}")
        print(f"FUTU-QUANT Multi-Strategy Live Trading")
        print(f"Mode:       {'DRY-RUN' if self.dry_run else '*** REAL MONEY ***'}")
        print(f"Symbols:    {sym_list}")
        print(f"Strategies: {n_strats} swing (intraday disabled, all failed 8yr validation)")
        print(f"Risk Mgmt:  QQQ SMA200 filter + VIX adaptive sizing + Dual Momentum rotation")
        print(f"Capital:    ${self.config['account']['initial_capital']:,.0f}")
        print(f"PDT:        {self.pdt.remaining_day_trades()}/{self.pdt.max_day_trades} remaining")
        print(f"Swing pos:  {self.holding_symbol or 'FLAT'}")
        print(f"Interval:   {interval}s")
        print(f"{'=' * 60}")
        print("Press Ctrl+C to stop.\n")

        last_eval_date = None

        while True:
            try:
                now = datetime.now()
                today_str = now.strftime("%Y-%m-%d")

                # Daily swing evaluation (once per day)
                if today_str != last_eval_date:
                    self.logger.info(f"--- Daily swing evaluation for {today_str} ---")
                    self.run_once()
                    last_eval_date = today_str

                if self._is_market_hours():
                    self.run_intraday_tick()

                # Monitor swing position hard stop
                if self.holding_symbol and self.holding_qty > 0:
                    price = self.get_current_price(self.holding_symbol)
                    if price:
                        pnl_pct = (price / self.holding_avg_price - 1) * 100
                        self.logger.info(
                            f"[MONITOR] {self.holding_symbol} ${price:.2f} "
                            f"PnL={pnl_pct:+.1f}% x{self.holding_qty} ({self.holding_strategy})"
                        )
                        hard_stop = self.config["risk"].get("hard_stop_pct", 0.08)
                        if pnl_pct < -hard_stop * 100:
                            self.logger.warning(
                                f"[HARD STOP] {self.holding_symbol} loss {pnl_pct:.2f}% "
                                f"exceeds {hard_stop * 100:.0f}%. SELLING."
                            )
                            self.execute_sell(price, is_intraday=False)

                time.sleep(interval)

            except KeyboardInterrupt:
                self.logger.info("Shutdown requested")
                break
            except Exception as e:
                self.logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="FUTU-QUANT Multi-Strategy Live Trading")
    parser.add_argument("--dry-run", action="store_true", help="Test without real orders")
    parser.add_argument("--once", action="store_true", help="Evaluate once and exit")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds (default 5min)")
    parser.add_argument("--config", default="config/live.yaml", help="Config file path")
    args = parser.parse_args()

    trader = MultiStrategyTrader(config_path=args.config, dry_run=args.dry_run)

    if not trader.connect():
        print("ERROR: Failed to connect. Is FutuOpenD running?")
        sys.exit(1)

    try:
        if args.once:
            trader.run_once()
        else:
            trader.run_loop(interval=args.interval)
    finally:
        trader.disconnect()


if __name__ == "__main__":
    main()
