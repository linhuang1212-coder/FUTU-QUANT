# Strategy Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the FUTU-QUANT 4-ETF rotation trading system across 6 dimensions: dynamic position sizing, signal quality filtering, UPRO/TECL parameter optimization, smart rotation logic, trailing stop exits, and cash yield layer.

**Architecture:** Each optimization is implemented as a self-contained module change with a dedicated backtest validation script. Changes are validated against the existing baseline (10yr Sharpe 0.881, CAGR 25.1%, MaxDD -39.0%) using the vectorized backtest framework. All changes are additive — each can be toggled on/off via `config/live.yaml`.

**Tech Stack:** Python 3.11+, pandas, numpy, existing strategy framework (`strategy/base.py`), existing backtest framework (`run_full_system_backtest.py`), existing risk framework (`risk/vol_target.py`)

**Baseline Performance (Rotation + SMA200, LIVE config):**

| Period | Sharpe | CAGR | MaxDD | Trades | $3000-> |
|--------|--------|------|-------|--------|---------|
| 10yr   | 0.881  | 25.1%| -39.0%| 1207   | $27,690 |
| 5yr    | 0.838  | 23.3%| -38.6%| 503    | $8,588  |
| 3yr    | 1.136  | 37.2%| -34.9%| 345    | $7,820  |
| 1yr    | 2.219  | 96.6%| -15.5%| 113    | $6,076  |

---

### Task 1: Dynamic Position Sizing Based on Signal Strength & Momentum Score

**Problem:** Fixed 72% allocation regardless of signal quality or momentum conviction. TQQQ single-symbol with 95% allocation achieves Sharpe 1.184 vs system's 0.881.

**Files:**
- Modify: `run_live.py` — `execute_buy()` and `_vix_adaptive_allocation()` methods
- Modify: `run_full_system_backtest.py` — `bt_rotation()` function
- Modify: `config/live.yaml` — add `position_sizing` config section

- [ ] **Step 1: Add position_sizing config to live.yaml**

Add after the `rotation` section in `config/live.yaml`:

```yaml
position_sizing:
  enabled: true
  base_allocation: 0.72
  # Signal strength scaling: allocation = base + (strength/100) * strength_bonus
  strength_bonus: 0.20       # max +20% allocation for strength=100 signals
  # Momentum score scaling: allocation *= (1 + momentum_score * mom_bonus)
  momentum_bonus: 0.15       # max +15% for top momentum
  max_allocation: 0.95
  min_allocation: 0.40
```

- [ ] **Step 2: Implement dynamic allocation in run_live.py**

Replace the `execute_buy` method's fixed allocation logic. Modify `execute_buy()` to accept `signal_strength` and `momentum_score` parameters:

```python
def _compute_dynamic_allocation(
    self, signal_strength: float = 70.0, momentum_score: float = 0.0
) -> float:
    """Compute allocation based on signal strength and momentum score.
    Returns fraction 0.0 - 0.95."""
    ps_cfg = self.config.get("position_sizing", {})
    if not ps_cfg.get("enabled", False):
        return self._vix_adaptive_allocation(0.95)

    base = ps_cfg.get("base_allocation", 0.72)
    str_bonus = ps_cfg.get("strength_bonus", 0.20)
    mom_bonus = ps_cfg.get("momentum_bonus", 0.15)
    max_alloc = ps_cfg.get("max_allocation", 0.95)
    min_alloc = ps_cfg.get("min_allocation", 0.40)

    # Scale by signal strength (0-100 -> 0-1)
    alloc = base + (signal_strength / 100.0) * str_bonus

    # Scale by momentum score (can be negative)
    if momentum_score > 0:
        alloc *= (1.0 + momentum_score * mom_bonus)

    alloc = max(min_alloc, min(alloc, max_alloc))

    # Apply VIX/vol regime scaling on top
    regime_scale = self._regime.position_scale if self._regime else 1.0
    final = alloc * regime_scale
    return max(min_alloc * 0.5, min(final, max_alloc))
```

Update `execute_buy()` signature to accept these new params and use `_compute_dynamic_allocation`.

Update the call site in `run_once()` where `execute_buy` is called to pass `best['signal'].strength` and the momentum score from `_momentum_rotation_rank`.

- [ ] **Step 3: Update backtest to use dynamic allocation**

In `run_full_system_backtest.py`, update `bt_rotation()` to use signal-strength-based allocation instead of fixed 0.72:

```python
# In the Buy section of bt_rotation, replace:
#   q = int(cap * 0.72 / closes[best])
# with:
    sig_str = all_buy[best][idx]
    base_alloc = 0.72
    str_bonus = 0.20
    alloc = base_alloc + (min(sig_str, 100) / 100.0) * str_bonus
    mom_sc = mom.get(best, 0)
    if mom_sc > 0:
        alloc *= (1.0 + mom_sc * 0.15)
    alloc = max(0.40, min(alloc, 0.95))
    q = int(cap * alloc / closes[best])
```

- [ ] **Step 4: Run backtest comparison**

Run: `python run_full_system_backtest.py`
Compare "Rotation + SMA200 (LIVE config)" numbers against the baseline.
Expected: CAGR improvement of 3-8% with Sharpe maintained or improved.

---

### Task 2: Signal Quality Filtering (Reduce False Signals)

**Problem:** TQQQ generates 1377 BUY signals over 2511 days (avg every 1.8 days). Current win rate ~31-35%. Too many low-quality entries.

**Files:**
- Create: `strategy/signal_filter.py` — signal confirmation and quality filter
- Modify: `run_live.py` — integrate signal filter in `_collect_swing_signals()`
- Modify: `run_full_system_backtest.py` — integrate signal filter in `precompute_all_signals()`
- Modify: `config/live.yaml` — add `signal_filter` config section

- [ ] **Step 1: Add signal_filter config to live.yaml**

Add after the `position_sizing` section:

```yaml
signal_filter:
  enabled: true
  min_strength: 60.0           # Reject signals below this strength
  confirmation_days: 1         # Require N consecutive days of same-direction signal
  min_strategies_agree: 2      # At least N strategies must agree on direction
  adx_entry_min: 20.0          # Minimum ADX for trend strategy entries
  volume_confirm: true         # Require above-average volume on signal day
```

- [ ] **Step 2: Create signal_filter.py**

Create `strategy/signal_filter.py`:

```python
"""Signal quality filter to reduce false entries.

Applies multiple confirmation checks to raw strategy signals:
1. Minimum strength threshold
2. Multi-strategy agreement (N strategies must agree)
3. Volume confirmation (above 20-day average)
4. ADX minimum for trend strategies
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class FilterResult:
    passed: bool
    reason: str
    adjusted_strength: float


TREND_STRATEGIES = {"momentum", "breakout"}


class SignalFilter:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.min_strength = config.get("min_strength", 60.0)
        self.confirmation_days = config.get("confirmation_days", 1)
        self.min_strategies_agree = config.get("min_strategies_agree", 2)
        self.adx_entry_min = config.get("adx_entry_min", 20.0)
        self.volume_confirm = config.get("volume_confirm", True)
        # Track recent signals for confirmation: {symbol: [direction_str, ...]}
        self._recent_signals: dict[str, list[str]] = {}

    def filter_signals(
        self,
        signals: list[dict],
        adx: float = 0.0,
        df_map: Optional[dict[str, pd.DataFrame]] = None,
    ) -> list[dict]:
        """Filter a batch of signals for one evaluation cycle.
        
        signals: list of dicts with keys: symbol, signal, strategy_name, score, layer
        adx: current ADX value
        df_map: {symbol: DataFrame} for volume checks
        """
        if not self.enabled:
            return signals

        # Group by (symbol, direction)
        from collections import defaultdict
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for s in signals:
            key = (s["symbol"], s["signal"].direction.value)
            groups[key].append(s)

        filtered = []
        for (sym, direction), group in groups.items():
            # Check 1: minimum strength
            strong_signals = [s for s in group if s["signal"].strength >= self.min_strength]
            if not strong_signals:
                continue

            # Check 2: multi-strategy agreement
            unique_strats = set(s["strategy_name"] for s in strong_signals)
            if direction == "BUY" and len(unique_strats) < self.min_strategies_agree:
                continue

            # Check 3: ADX minimum for trend strategies
            if direction == "BUY":
                trend_only = all(s["strategy_name"] in TREND_STRATEGIES for s in strong_signals)
                if trend_only and adx < self.adx_entry_min:
                    continue

            # Check 4: Volume confirmation
            if self.volume_confirm and df_map and sym in df_map:
                df = df_map[sym]
                if len(df) >= 20:
                    vol_avg = df["volume"].rolling(20).mean().iloc[-1]
                    cur_vol = df["volume"].iloc[-1]
                    if cur_vol < vol_avg * 0.8:
                        continue

            # Boost strength based on agreement count
            best = max(strong_signals, key=lambda s: s["score"])
            agreement_bonus = (len(unique_strats) - 1) * 5.0
            best["signal"].strength = min(best["signal"].strength + agreement_bonus, 100.0)
            best["score"] = best["sharpe_weight"] * best["signal"].strength
            filtered.append(best)

        return filtered

    def filter_signals_vectorized(
        self,
        buy_scores: dict[str, "np.ndarray"],
        sell_scores: dict[str, "np.ndarray"],
        all_ind: dict[str, pd.DataFrame],
        symbols: list[str],
    ) -> tuple[dict[str, "np.ndarray"], dict[str, "np.ndarray"]]:
        """Vectorized version for backtest: zero out weak signals."""
        import numpy as np

        if not self.enabled:
            return buy_scores, sell_scores

        new_buy = {}
        new_sell = {}
        for sym in symbols:
            bs = buy_scores[sym].copy()
            ss = sell_scores[sym].copy()

            # Strength threshold: zero out signals below min_strength
            # Score = weight * strength; assume avg weight ~1.0
            bs[bs < self.min_strength * 0.8] = 0
            ss[ss < self.min_strength * 0.8] = 0

            # Volume confirmation
            if self.volume_confirm and sym in all_ind:
                df = all_ind[sym]
                vol_avg = df["volume"].rolling(20).mean().values
                cur_vol = df["volume"].values
                low_vol = cur_vol < vol_avg * 0.8
                low_vol = low_vol | pd.isna(vol_avg)
                bs[low_vol] = 0

            new_buy[sym] = bs
            new_sell[sym] = ss

        return new_buy, new_sell
```

- [ ] **Step 3: Integrate signal filter in run_live.py**

In `__init__`, after loading rotation_cfg:
```python
from strategy.signal_filter import SignalFilter
sf_cfg = self.config.get("signal_filter", {})
self.signal_filter = SignalFilter(sf_cfg)
```

In `_collect_swing_signals()`, before returning `results`, add:
```python
if self.signal_filter.enabled:
    adx_val = self._regime.adx_value if self._regime else 0.0
    results = self.signal_filter.filter_signals(
        results, adx=adx_val, df_map=None
    )
```

- [ ] **Step 4: Integrate signal filter in backtest**

In `run_full_system_backtest.py`, after precomputing all signals, apply the vectorized filter:

```python
from strategy.signal_filter import SignalFilter
sf = SignalFilter({
    "enabled": True, "min_strength": 60.0,
    "min_strategies_agree": 2, "volume_confirm": True,
    "adx_entry_min": 20.0,
})
all_buy, all_sell = sf.filter_signals_vectorized(
    all_buy, all_sell, all_ind, POOL
)
```

- [ ] **Step 5: Run backtest and compare**

Run: `python run_full_system_backtest.py`
Expected: Fewer trades (from ~1207 to ~400-600), higher win rate (35% -> 45%+), Sharpe improvement.

---

### Task 3: UPRO/TECL Walk-Forward Parameter Optimization

**Problem:** UPRO 5yr Sharpe only 0.401 (CAGR 8%), TECL 0.304 (CAGR 4.1%). Both use default parameters with `sharpe_weight: 1.0`, never optimized.

**Files:**
- Create: `run_optimize_upro_tecl.py` — dedicated optimization script
- Modify: `config/live.yaml` — update optimized params and sharpe_weights for UPRO/TECL

- [ ] **Step 1: Create optimization script**

Create `run_optimize_upro_tecl.py` that runs Walk-Forward optimization on UPRO and TECL. Uses the same approach as was done for TQQQ/SOXL: grid search over key parameters, 8 windows of 3yr in-sample + 1yr out-of-sample:

```python
"""Walk-Forward optimization for UPRO and TECL.

For each strategy (breakout, mean_reversion, multi_factor), search key params
over a grid, evaluate each on 8 rolling windows (3yr train + 1yr test).
Select params with best average OOS Sharpe and >=60% window consistency.
"""

import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from strategy.breakout import BreakoutStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.multi_factor import MultiFactorStrategy
from data.indicators import TechnicalIndicators

DATA_DIR = Path("data_store/market_data")
TARGETS = ["UPRO", "TECL"]
INITIAL_CAPITAL = 3000.0
HARD_STOP = -0.08
WIN = 60


def load(sym):
    return pd.read_csv(
        DATA_DIR / f"{sym}_daily.csv", parse_dates=["time_key"]
    ).sort_values("time_key").reset_index(drop=True)


def add_indicators(df):
    o = df.copy()
    for p in (5, 8, 10, 14, 15, 20):
        o = TechnicalIndicators.add_ma(o, p)
        o = TechnicalIndicators.add_ema(o, p)
    for p in (5, 7, 10, 14):
        o = TechnicalIndicators.add_rsi(o, p)
    for bp, bs in ((15, 2.0), (20, 2.0)):
        o = TechnicalIndicators.add_bollinger(o, bp, bs)
    o = TechnicalIndicators.add_atr(o, 14)
    o = TechnicalIndicators.add_macd(o, 12, 26, 9)
    return o


def backtest_single(df_ind, strat, start_idx, end_idx):
    """Run strategy on a slice, return Sharpe."""
    closes = df_ind["close"].values
    cap = INITIAL_CAPITAL
    eq = [cap]
    pos = None
    for i in range(max(start_idx, WIN), end_idx):
        p = closes[i]
        window = df_ind.iloc[i - WIN:i + 1]
        if pos and (p / pos[0] - 1) <= HARD_STOP:
            cap += pos[1] * (p - pos[0])
            pos = None
        try:
            sig = strat.on_bar("TEST", window)
        except Exception:
            sig = None
        if pos and sig and sig.direction.value == "SELL":
            cap += pos[1] * (p - pos[0])
            pos = None
        elif not pos and sig and sig.direction.value == "BUY":
            q = int(cap * 0.95 / p)
            if q > 0:
                pos = (p, q)
        eq.append(cap + pos[1] * (p - pos[0]) if pos else cap)

    v = np.array(eq)
    if len(v) < 20:
        return 0.0
    r = np.diff(v) / v[:-1]
    r = r[np.isfinite(r)]
    return float((np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 0 else 0)


PARAM_GRIDS = {
    "breakout": {
        "cls": BreakoutStrategy,
        "grid": {
            "lookback_period": [8, 10, 15, 20],
            "volume_ratio_threshold": [1.0, 1.2, 1.5],
            "atr_breakout_multiplier": [1.0, 1.5, 2.0],
        },
    },
    "mean_reversion": {
        "cls": MeanReversionStrategy,
        "grid": {
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_period": [10, 14],
            "rsi_oversold": [20, 25, 30],
            "rsi_overbought": [70, 75, 80],
        },
    },
    "multi_factor": {
        "cls": MultiFactorStrategy,
        "grid": {
            "fast_ma_period": [5, 8, 10],
            "slow_ma_period": [15, 20, 25],
            "rsi_period": [10, 14],
            "ema_period": [15, 20],
            "buy_threshold": [3, 4],
            "sell_threshold": [3, 4],
        },
    },
}

TRAIN_YEARS = 3
TEST_YEARS = 1
DAYS_PER_YEAR = 252


def walk_forward(sym, df_ind, strat_name, cls, param_combos):
    """Walk-Forward: 8 windows, each 3yr train + 1yr test."""
    n = len(df_ind)
    window_size = (TRAIN_YEARS + TEST_YEARS) * DAYS_PER_YEAR
    step = TEST_YEARS * DAYS_PER_YEAR
    n_windows = min(8, (n - window_size) // step + 1)

    if n_windows < 3:
        print(f"  WARNING: Only {n_windows} windows for {sym}/{strat_name}")
        return None, 0, 0

    best_params = None
    best_avg_oos = -999
    best_consistency = 0

    total = len(param_combos)
    for pi, params in enumerate(param_combos):
        if pi % 50 == 0:
            print(f"    {strat_name}: combo {pi}/{total}...", flush=True)

        oos_sharpes = []
        for wi in range(n_windows):
            train_start = n - window_size - (n_windows - 1 - wi) * step
            train_end = train_start + TRAIN_YEARS * DAYS_PER_YEAR
            test_end = train_end + TEST_YEARS * DAYS_PER_YEAR
            if train_start < WIN or test_end > n:
                continue

            strat = cls(params=dict(params))
            is_sharpe = backtest_single(df_ind, strat, train_start, train_end)
            if is_sharpe < 0.3:
                oos_sharpes.append(0)
                continue
            oos_sharpe = backtest_single(df_ind, strat, train_end, test_end)
            oos_sharpes.append(oos_sharpe)

        if len(oos_sharpes) < 3:
            continue
        avg_oos = np.mean(oos_sharpes)
        consistency = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)

        if avg_oos > best_avg_oos and consistency >= 0.6:
            best_avg_oos = avg_oos
            best_params = dict(params)
            best_consistency = consistency

    return best_params, best_avg_oos, best_consistency


def main():
    for sym in TARGETS:
        print(f"\n{'='*70}")
        print(f"  Walk-Forward Optimization: {sym}")
        print(f"{'='*70}")
        df_raw = load(sym)
        df_ind = add_indicators(df_raw)
        print(f"  Data: {len(df_raw)} bars ({df_raw['time_key'].iloc[0].date()} ~ {df_raw['time_key'].iloc[-1].date()})")

        for sname, scfg in PARAM_GRIDS.items():
            cls = scfg["cls"]
            grid = scfg["grid"]
            keys = list(grid.keys())
            values = list(grid.values())
            combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
            print(f"\n  Strategy: {sname} ({len(combos)} param combos)")

            best_params, avg_oos, consistency = walk_forward(
                sym, df_ind, sname, cls, combos
            )
            if best_params:
                print(f"  >>> BEST: avg OOS Sharpe={avg_oos:.3f}, consistency={consistency:.0%}")
                print(f"  >>> Params: {best_params}")
            else:
                print(f"  >>> No params passed consistency filter (>=60%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run optimization**

Run: `python run_optimize_upro_tecl.py`
This will take ~10-30 minutes. Record the optimal params for each strategy.

- [ ] **Step 3: Update live.yaml with optimized params**

Replace the UPRO and TECL strategy params in `config/live.yaml` with the Walk-Forward optimized values. Update `sharpe_weight` to the average OOS Sharpe for each strategy.

- [ ] **Step 4: Re-run full backtest to validate**

Run: `python run_full_system_backtest.py`
Expected: UPRO per-symbol Sharpe improves from 0.4 to 0.7+, TECL from 0.3 to 0.6+.

---

### Task 4: Smart Rotation Logic (Risk-Adjusted Momentum + Hysteresis)

**Problem:** Current rotation uses raw momentum only. Frequent switching between symbols. High-volatility symbols get overweighted. No hysteresis to prevent whipsaw rotations.

**Files:**
- Modify: `run_live.py` — `_momentum_rotation_rank()` method
- Modify: `run_full_system_backtest.py` — momentum ranking in `bt_rotation()`
- Modify: `config/live.yaml` — add rotation enhancements config

- [ ] **Step 1: Add smart rotation config to live.yaml**

Enhance the existing `rotation` section:

```yaml
rotation:
  enabled: true
  candidate_count: 2
  momentum_weights:
    mom_1m: 0.5
    mom_3m: 0.5
  min_momentum: 0.0
  rerank_interval_days: 1
  # New: risk-adjusted momentum
  risk_adjust: true            # Divide momentum by realized volatility
  vol_lookback: 21             # Days for realized vol calculation
  # New: hysteresis to prevent whipsaw
  hysteresis_pct: 0.03         # Current holding must underperform by >3% to trigger switch
  # New: dynamic candidate count
  dynamic_candidates: true
  high_trend_adx: 30           # ADX above this -> concentrate (1 candidate)
  low_trend_adx: 20            # ADX below this -> diversify (3 candidates, capped at pool size)
```

- [ ] **Step 2: Implement enhanced rotation in run_live.py**

Replace `_momentum_rotation_rank()`:

```python
def _momentum_rotation_rank(self) -> list[str]:
    """Rank all pool symbols by risk-adjusted momentum with hysteresis.
    Returns top-N candidates with positive momentum."""
    cfg = self.rotation_cfg
    w1m = cfg.get("momentum_weights", {}).get("mom_1m", 0.5)
    w3m = cfg.get("momentum_weights", {}).get("mom_3m", 0.5)
    min_mom = cfg.get("min_momentum", 0.0)
    risk_adjust = cfg.get("risk_adjust", True)
    vol_lookback = cfg.get("vol_lookback", 21)
    hysteresis = cfg.get("hysteresis_pct", 0.03)

    # Dynamic candidate count based on ADX
    if cfg.get("dynamic_candidates", False) and self._regime:
        adx = self._regime.adx_value
        if adx >= cfg.get("high_trend_adx", 30):
            top_n = 1
        elif adx <= cfg.get("low_trend_adx", 20):
            top_n = min(3, len(self._all_symbols()))
        else:
            top_n = cfg.get("candidate_count", 2)
    else:
        top_n = cfg.get("candidate_count", 2)

    scores: dict[str, float] = {}
    for sym in self._all_symbols():
        df = self.get_daily_kline(sym, 80)
        if df is None or len(df) < 63:
            continue
        close = df["close"].values
        mom_1m = close[-1] / close[-21] - 1 if len(close) >= 21 else 0
        mom_3m = close[-1] / close[-63] - 1 if len(close) >= 63 else 0
        raw_mom = w1m * mom_1m + w3m * mom_3m

        if risk_adjust and len(close) >= vol_lookback:
            rets = pd.Series(close).pct_change().dropna().tail(vol_lookback)
            vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 1.0
            scores[sym] = raw_mom / max(vol, 0.01)
        else:
            scores[sym] = raw_mom

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Hysteresis: if we hold a symbol, it needs to drop significantly in rank
    if self.holding_symbol and self.holding_symbol in scores:
        held_score = scores[self.holding_symbol]
        top_score = ranked[0][1] if ranked else 0
        if held_score > 0 and (top_score - held_score) < hysteresis:
            # Keep current holding in candidates
            candidates = [self.holding_symbol]
            for sym, sc in ranked:
                if sym != self.holding_symbol and sc > min_mom:
                    candidates.append(sym)
                    if len(candidates) >= top_n:
                        break
            for i, (sym, score) in enumerate(ranked):
                tag = "TOP" if sym in candidates else "skip"
                if sym == self.holding_symbol:
                    tag = "HOLD (hysteresis)"
                self.logger.info(f"[ROTATION] #{i+1} {sym}: score={score:+.3f} ({tag})")
            return candidates

    for i, (sym, score) in enumerate(ranked):
        tag = "TOP" if i < top_n and score > min_mom else "skip"
        self.logger.info(f"[ROTATION] #{i+1} {sym}: score={score:+.3f} ({tag})")

    candidates = [sym for sym, score in ranked[:top_n] if score > min_mom]
    if not candidates:
        self.logger.info("[ROTATION] All momentum negative -> stay cash")
    return candidates
```

- [ ] **Step 3: Update backtest rotation logic**

In `run_full_system_backtest.py`, update the momentum ranking section in `bt_rotation()`:

```python
# Replace the simple momentum block with risk-adjusted version:
mom = {}
for s in POOL:
    before = all_raw[s][all_raw[s].index <= idx]
    if len(before) < 63:
        continue
    c = before["close"].values
    raw = 0.5 * (c[-1] / c[-21] - 1) + 0.5 * (c[-1] / c[-63] - 1)
    rets = pd.Series(c).pct_change().dropna().tail(21)
    vol = rets.std() * np.sqrt(252) if len(rets) > 5 else 1.0
    mom[s] = raw / max(vol, 0.01)

# Hysteresis
if hold:
    held_s = mom.get(hold[0], -999)
    top_s = max(mom.values()) if mom else 0
    if held_s > 0 and (top_s - held_s) < 0.03:
        cands = {hold[0]}
        for s2, sc2 in sorted(mom.items(), key=lambda x: -x[1]):
            if s2 != hold[0] and sc2 > 0:
                cands.add(s2)
                if len(cands) >= 2:
                    break
    else:
        cands = set(s for s, sc in sorted(mom.items(), key=lambda x: -x[1])[:2] if sc > 0)
else:
    cands = set(s for s, sc in sorted(mom.items(), key=lambda x: -x[1])[:2] if sc > 0)
```

- [ ] **Step 4: Run backtest and compare**

Run: `python run_full_system_backtest.py`
Expected: Fewer rotation switches, more stable returns, Sharpe improvement.

---

### Task 5: Trailing Stop Exit Strategy

**Problem:** Only hard stop at -8%. No profit locking mechanism. Large winners can reverse entirely before getting a SELL signal.

**Files:**
- Create: `risk/trailing_stop.py` — trailing stop logic
- Modify: `run_live.py` — integrate trailing stop in `run_loop()` monitor section
- Modify: `run_full_system_backtest.py` — add trailing stop to simulation
- Modify: `config/live.yaml` — add `trailing_stop` config (enhance existing)

- [ ] **Step 1: Enhance trailing_stop config in live.yaml**

Replace the existing `trailing_stop` under `risk`:

```yaml
risk:
  # ... existing fields ...
  trailing_stop:
    enabled: true
    # Tier 1: activate after +5% profit, trail 3%
    activate_pct: 0.05
    trail_pct: 0.03
    # Tier 2: tighten after +15% profit, trail 2%
    tier2_activate_pct: 0.15
    tier2_trail_pct: 0.02
    # ATR-based dynamic stop (overrides fixed % if more favorable)
    atr_enabled: true
    atr_multiplier: 2.5        # Stop at highest_price - 2.5*ATR
```

- [ ] **Step 2: Create risk/trailing_stop.py**

```python
"""Tiered trailing stop with ATR-based dynamic adjustment.

Tracks highest price since entry. When profit exceeds tier thresholds,
activates trailing stops that lock in gains. ATR-based stop provides
a volatility-aware alternative.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrailingStopState:
    entry_price: float
    highest_price: float
    active_tier: int  # 0=inactive, 1=tier1, 2=tier2
    stop_price: float

    @property
    def pnl_from_peak(self) -> float:
        if self.highest_price <= 0:
            return 0.0
        return (self.highest_price - self.stop_price) / self.entry_price


class TrailingStopManager:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.activate_pct = config.get("activate_pct", 0.05)
        self.trail_pct = config.get("trail_pct", 0.03)
        self.tier2_activate_pct = config.get("tier2_activate_pct", 0.15)
        self.tier2_trail_pct = config.get("tier2_trail_pct", 0.02)
        self.atr_enabled = config.get("atr_enabled", True)
        self.atr_multiplier = config.get("atr_multiplier", 2.5)
        self._states: dict[str, TrailingStopState] = {}

    def on_entry(self, symbol: str, entry_price: float):
        self._states[symbol] = TrailingStopState(
            entry_price=entry_price,
            highest_price=entry_price,
            active_tier=0,
            stop_price=0.0,
        )

    def on_exit(self, symbol: str):
        self._states.pop(symbol, None)

    def update(
        self, symbol: str, current_price: float, current_atr: float = 0.0
    ) -> Optional[str]:
        """Update trailing stop state. Returns exit reason if stop triggered, else None."""
        if not self.enabled or symbol not in self._states:
            return None

        state = self._states[symbol]

        # Update highest price
        if current_price > state.highest_price:
            state.highest_price = current_price

        pnl_pct = (state.highest_price / state.entry_price) - 1.0

        # Determine active tier
        if pnl_pct >= self.tier2_activate_pct:
            state.active_tier = 2
            trail = self.tier2_trail_pct
        elif pnl_pct >= self.activate_pct:
            state.active_tier = 1
            trail = self.trail_pct
        else:
            state.active_tier = 0
            return None

        # Fixed percentage stop
        fixed_stop = state.highest_price * (1 - trail)

        # ATR-based stop (if available and more favorable)
        if self.atr_enabled and current_atr > 0:
            atr_stop = state.highest_price - self.atr_multiplier * current_atr
            stop = max(fixed_stop, atr_stop)
        else:
            stop = fixed_stop

        # Ensure stop never moves down
        state.stop_price = max(state.stop_price, stop)

        # Check if triggered
        if current_price <= state.stop_price:
            pnl = (current_price / state.entry_price - 1) * 100
            reason = (
                f"Trailing stop T{state.active_tier} triggered: "
                f"peak=${state.highest_price:.2f} stop=${state.stop_price:.2f} "
                f"PnL={pnl:+.1f}%"
            )
            return reason

        return None

    def get_state(self, symbol: str) -> Optional[TrailingStopState]:
        return self._states.get(symbol)
```

- [ ] **Step 3: Integrate trailing stop in run_live.py**

In `__init__`, add:
```python
from risk.trailing_stop import TrailingStopManager
ts_cfg = self.config.get("risk", {}).get("trailing_stop", {})
self.trailing_stop = TrailingStopManager(ts_cfg)
```

In `execute_buy()`, after confirming the buy:
```python
self.trailing_stop.on_entry(symbol, price)
```

In `execute_sell()`, after confirming the sell:
```python
self.trailing_stop.on_exit(symbol)
```

In `run_loop()`, in the monitoring section where hard stop is checked, add trailing stop check before the hard stop:

```python
# In the monitoring block, after getting price:
ts_reason = self.trailing_stop.update(
    self.holding_symbol, price, current_atr=0.0
)
if ts_reason:
    self.logger.info(f"[TRAILING STOP] {ts_reason}")
    self.execute_sell(price, is_intraday=False)
    continue
```

Also fetch ATR for the trailing stop if available:
```python
if self.holding_symbol and self.holding_qty > 0:
    price = self.get_current_price(self.holding_symbol)
    if price:
        # Get ATR for trailing stop
        atr_val = 0.0
        df_ts = self.get_daily_kline(self.holding_symbol, 20)
        if df_ts is not None and len(df_ts) >= 14:
            df_ts = TechnicalIndicators.add_atr(df_ts, 14)
            atr_col = df_ts.get("atr_14")
            if atr_col is not None and not pd.isna(df_ts["atr_14"].iloc[-1]):
                atr_val = float(df_ts["atr_14"].iloc[-1])

        ts_reason = self.trailing_stop.update(self.holding_symbol, price, atr_val)
        if ts_reason:
            self.logger.info(f"[TRAILING STOP] {ts_reason}")
            self.execute_sell(price, is_intraday=False)
        else:
            # Existing hard stop logic...
```

- [ ] **Step 4: Add trailing stop to backtest**

In `run_full_system_backtest.py`, update `bt_rotation()` to track highest price and apply trailing stop:

```python
# Add tracking variables at the start of bt_rotation:
highest_price = 0.0
ts_activate = 0.05
ts_trail = 0.03
ts_tier2_activate = 0.15
ts_tier2_trail = 0.02

# After hard stop check and before sell signal check:
if hold and closes.get(hold[0]):
    cur_p = closes[hold[0]]
    highest_price = max(highest_price, cur_p)
    pnl_from_entry = (highest_price / hold[1]) - 1.0

    if pnl_from_entry >= ts_tier2_activate:
        ts_stop = highest_price * (1 - ts_tier2_trail)
    elif pnl_from_entry >= ts_activate:
        ts_stop = highest_price * (1 - ts_trail)
    else:
        ts_stop = 0

    if ts_stop > 0 and cur_p <= ts_stop:
        cap += hold[2] * (cur_p - hold[1])
        hold = None
        highest_price = 0
        tr += 1

# When entering a new position:
# highest_price = closes[best]
```

- [ ] **Step 5: Run backtest and compare**

Run: `python run_full_system_backtest.py`
Expected: MaxDD improvement (from -39% to -30% or less), profits better preserved.

---

### Task 6: Cash Yield Layer (Short-Term Treasury During Flat Periods)

**Problem:** System is flat ~25% of the time. Cash earns 0% during these periods.

**Files:**
- Modify: `run_full_system_backtest.py` — add cash yield calculation
- Modify: `run_live.py` — add logging/tracking of cash yield
- Modify: `config/live.yaml` — add `cash_yield` config

- [ ] **Step 1: Add cash_yield config to live.yaml**

```yaml
cash_yield:
  enabled: true
  # Annual yield of cash alternative (BIL/SHY short-term treasury)
  annual_yield_pct: 4.5
  # Symbol for cash alternative (informational, not auto-traded)
  cash_etf: "US.BIL"
```

- [ ] **Step 2: Update backtest to include cash yield**

In `run_full_system_backtest.py`, modify `bt_rotation()` to accrue interest on flat days:

```python
# At the top of bt_rotation, add:
daily_yield = (1 + 0.045) ** (1/252) - 1  # 4.5% annualized

# In the daily loop, when not holding:
if hold and closes.get(hold[0]):
    eq.append(cap + hold[2] * (closes[hold[0]] - hold[1]))
else:
    cap *= (1 + daily_yield)  # Cash earns yield
    eq.append(cap)
```

- [ ] **Step 3: Add cash yield tracking in run_live.py**

Add a simple logger line in `run_once()` when staying flat to remind the operator about opportunity cost:

```python
elif self.holding_symbol is None:
    cash_cfg = self.config.get("cash_yield", {})
    if cash_cfg.get("enabled", False):
        self.logger.info(
            f"No swing BUY signals. Staying flat. "
            f"Consider {cash_cfg.get('cash_etf', 'BIL')} for idle cash "
            f"({cash_cfg.get('annual_yield_pct', 4.5)}% annual yield)."
        )
    else:
        self.logger.info("No swing BUY signals. Staying flat.")
```

- [ ] **Step 4: Run backtest and compare**

Run: `python run_full_system_backtest.py`
Expected: ~1-2% annual CAGR improvement from cash yield during flat periods.

---

### Task 7: Combined Validation & Final Comparison

**Purpose:** Run the complete system with ALL optimizations enabled, compare against baseline across all time segments.

**Files:**
- Modify: `run_full_system_backtest.py` — ensure all optimizations are active
- Create: `run_optimization_report.py` — side-by-side comparison script

- [ ] **Step 1: Create comparison script**

Create `run_optimization_report.py` that runs the full backtest twice — once with original config, once with all optimizations — and prints a side-by-side comparison:

```python
"""Compare original vs optimized system performance."""

import numpy as np
import pandas as pd
from pathlib import Path
from run_full_system_backtest import (
    load, add_indicators, precompute_all_signals,
    bt_rotation, calc, POOL, INITIAL_CAPITAL
)

DATA_DIR = Path("data_store/market_data")

SEGMENTS = [
    ("10yr", "2016-05-01", "2026-04-17"),
    ("5yr",  "2021-04-01", "2026-04-17"),
    ("3yr",  "2023-04-01", "2026-04-17"),
    ("1yr",  "2025-04-01", "2026-04-17"),
]

# Baseline numbers from before optimization
BASELINE = {
    "10yr": {"sharpe": 0.881, "cagr": 25.1, "maxdd": -39.0, "final": 27690},
    "5yr":  {"sharpe": 0.838, "cagr": 23.3, "maxdd": -38.6, "final": 8588},
    "3yr":  {"sharpe": 1.136, "cagr": 37.2, "maxdd": -34.9, "final": 7820},
    "1yr":  {"sharpe": 2.219, "cagr": 96.6, "maxdd": -15.5, "final": 6076},
}


def main():
    print("=" * 90)
    print("  FUTU-QUANT Optimization Report: Baseline vs Optimized")
    print("=" * 90)

    print("\nLoading data & computing signals...")
    all_raw, all_ind = {}, {}
    for sym in POOL:
        all_raw[sym] = load(sym)
        all_ind[sym] = add_indicators(all_raw[sym])

    all_buy, all_sell = {}, {}
    for sym in POOL:
        b, s = precompute_all_signals(sym, all_ind[sym])
        all_buy[sym] = b
        all_sell[sym] = s

    print(f"\n{'Segment':<8} | {'Metric':<8} | {'Baseline':>10} | {'Optimized':>10} | {'Change':>10}")
    print("-" * 60)

    for seg_name, start, end in SEGMENTS:
        r = bt_rotation(all_raw, all_buy, all_sell, start, end, sma200=True)
        b = BASELINE[seg_name]

        for metric in ["sharpe", "cagr", "maxdd"]:
            bv = b[metric]
            ov = r[metric]
            delta = ov - bv
            sign = "+" if delta > 0 else ""
            print(f"{seg_name:<8} | {metric:<8} | {bv:>10.1f} | {ov:>10.1f} | {sign}{delta:>9.1f}")

        print("-" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full comparison**

Run: `python run_optimization_report.py`

Document the results. ALL of the following must improve over baseline for the optimization to be accepted:
- 10yr Sharpe >= 0.90 (was 0.881)
- 10yr MaxDD >= -38% (was -39%)
- 3yr Sharpe >= 1.15 (was 1.136)
- No segment should have Sharpe < 0.80

- [ ] **Step 3: Update live.yaml with final validated config**

If validation passes, ensure `config/live.yaml` has all optimizations enabled with the validated parameters.

- [ ] **Step 4: Test run_live.py with optimizations**

Run: `python run_live.py --once --dry-run`
Verify no errors, all new modules load correctly, and log output shows the new features (dynamic allocation, signal filter, trailing stop, smart rotation).
