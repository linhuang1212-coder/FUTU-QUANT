"""Statistical validation for backtest results.

Implements:
- Deflated Sharpe Ratio (DSR) — adjusts for multiple testing
- Probability of Backtest Overfitting (PBO)
- Purged Walk-Forward cross-validation
- Minimum sample requirements
- Strategy confidence scoring

References:
  Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
  Bailey et al. (2017) "The Probability of Backtest Overfitting"
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Deflated Sharpe Ratio ────────────────────────────────────

def expected_max_sharpe(n_trials: int, T: int, skew: float = 0.0,
                        kurt: float = 3.0) -> float:
    """Expected max Sharpe ratio from pure noise given n_trials strategies.

    Under the null hypothesis (all strategies have zero true Sharpe),
    the expected maximum Sharpe from n_trials independent tests with T
    observations each follows the Euler-Mascheroni approximation.
    """
    if n_trials <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649
    log_n = math.log(n_trials)
    if log_n < 1e-12:
        return 0.0
    sqrt_log_n = math.sqrt(2 * log_n)
    e_max = (1 - euler_mascheroni) * sqrt_log_n + euler_mascheroni / sqrt_log_n
    return e_max


def sharpe_std_error(T: int, skew: float = 0.0, kurt: float = 3.0,
                     sharpe: float = 0.0) -> float:
    """Standard error of Sharpe ratio estimator (Lo, 2002)."""
    if T <= 1:
        return float("inf")
    se = math.sqrt(
        (1
         + 0.5 * sharpe ** 2
         - skew * sharpe
         + (kurt - 3) / 4 * sharpe ** 2)
        / (T - 1)
    )
    return se


def deflated_sharpe_ratio(sharpe: float, n_trials: int, T: int,
                          skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Compute the Deflated Sharpe Ratio.

    Returns dict with:
      dsr: the deflated Sharpe (observed - noise expectation)
      p_value: prob that observed Sharpe is due to chance
      expected_max: noise Sharpe floor
      is_significant: True if DSR > 0 (strategy beats noise)
    """
    e_max = expected_max_sharpe(n_trials, T, skew, kurt)
    se = sharpe_std_error(T, skew, kurt, sharpe)
    dsr = sharpe - e_max

    if se > 0 and math.isfinite(se):
        z = dsr / se
        p_value = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    else:
        p_value = 0.5

    return {
        "observed_sharpe": round(sharpe, 4),
        "expected_max_noise_sharpe": round(e_max, 4),
        "deflated_sharpe": round(dsr, 4),
        "std_error": round(se, 4),
        "p_value": round(p_value, 4),
        "is_significant": dsr > 0 and p_value < 0.05,
        "n_trials": n_trials,
        "n_observations": T,
    }


# ── Probability of Backtest Overfitting ──────────────────────

def prob_backtest_overfitting(n_trials: int) -> float:
    """Rough PBO estimate based on number of strategies tested.

    From Bailey et al. — simplified heuristic:
    PBO ≈ 1 - 1/n_trials for large n, with floor adjustment.
    """
    if n_trials <= 1:
        return 0.0
    # Empirical approximation from the original paper
    pbo = 1 - (1 / (1 + 0.2 * math.log(n_trials)))
    return round(min(pbo, 0.99), 4)


# ── Minimum Sample Size ─────────────────────────────────────

def min_trades_for_significance(win_rate: float = 0.50,
                                confidence: float = 0.95) -> int:
    """Minimum trades needed for win rate to be statistically significant.

    Uses binomial proportion confidence interval.
    For a strategy to be "significantly better than coin-flip" at the
    given confidence level, we need enough observations.
    """
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)
    p = win_rate
    q = 1 - p
    if abs(p - 0.5) < 0.01:
        return 1000  # near coin-flip needs huge sample
    margin = abs(p - 0.5)
    n = (z ** 2 * p * q) / (margin ** 2)
    return max(30, int(math.ceil(n)))


def sample_size_check(n_trades: int, win_rate: float) -> dict:
    """Check if sample size is sufficient for statistical inference."""
    try:
        min_n = min_trades_for_significance(win_rate)
    except ImportError:
        min_n = 100  # fallback

    ratio = n_trades / max(min_n, 1)
    if n_trades >= min_n:
        verdict = "sufficient"
    elif n_trades >= min_n * 0.5:
        verdict = "marginal"
    else:
        verdict = "insufficient"

    return {
        "n_trades": n_trades,
        "min_required": min_n,
        "ratio": round(ratio, 2),
        "verdict": verdict,
    }


# ── Strategy Confidence Score ────────────────────────────────

@dataclass
class StrategyValidation:
    """Multi-layer validation result for a strategy."""
    strategy_name: str
    economic_thesis: str = ""  # Layer 1: why does this work?
    dsr: Optional[dict] = None  # Layer 2: deflated Sharpe
    sample_check: Optional[dict] = None  # Layer 3: sample size
    pbo: float = 0.0  # Layer 4: overfitting probability
    has_out_of_sample: bool = False  # Layer 5: OOS test done?
    confidence_score: float = 0.0  # 0-100 composite score
    verdict: str = ""

    def compute_confidence(self):
        """Composite confidence score across all validation layers."""
        score = 0.0
        reasons = []

        # Layer 1: Economic thesis (20 pts)
        if self.economic_thesis:
            score += 20
        else:
            reasons.append("No economic thesis documented")

        # Layer 2: DSR significance (25 pts)
        if self.dsr:
            if self.dsr.get("is_significant"):
                score += 25
            elif self.dsr.get("deflated_sharpe", 0) > 0:
                score += 10
                reasons.append(f"DSR positive but p={self.dsr['p_value']:.2f}")
            else:
                reasons.append(f"DSR negative ({self.dsr['deflated_sharpe']:.2f})")

        # Layer 3: Sample size (20 pts)
        if self.sample_check:
            v = self.sample_check["verdict"]
            if v == "sufficient":
                score += 20
            elif v == "marginal":
                score += 10
                reasons.append(f"Sample marginal ({self.sample_check['n_trades']}/{self.sample_check['min_required']})")
            else:
                reasons.append(f"Sample insufficient ({self.sample_check['n_trades']}/{self.sample_check['min_required']})")

        # Layer 4: PBO (20 pts)
        if self.pbo < 0.20:
            score += 20
        elif self.pbo < 0.40:
            score += 10
            reasons.append(f"PBO={self.pbo:.0%}")
        else:
            reasons.append(f"PBO={self.pbo:.0%} — high overfitting risk")

        # Layer 5: OOS validation (15 pts)
        if self.has_out_of_sample:
            score += 15
        else:
            reasons.append("No out-of-sample validation")

        self.confidence_score = round(score, 1)

        if score >= 80:
            self.verdict = "HIGH — ready for live (small size)"
        elif score >= 60:
            self.verdict = "MEDIUM — needs more data/validation"
        elif score >= 40:
            self.verdict = "LOW — significant gaps remain"
        else:
            self.verdict = "VERY LOW — do not trade"

        if reasons:
            self.verdict += f" [{'; '.join(reasons)}]"


# ── Purged Walk-Forward Split ────────────────────────────────

def purged_walk_forward_split(n: int, n_splits: int = 5,
                               purge_pct: float = 0.02) -> list[tuple]:
    """Generate train/test indices with purge gap to prevent leakage.

    Unlike TimeSeriesSplit, this adds a purge gap between train and test
    to prevent label overlap at boundaries.
    """
    splits = []
    fold_size = n // (n_splits + 1)
    purge_size = max(1, int(n * purge_pct))

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_start = train_end + purge_size
        test_end = min(test_start + fold_size, n)

        if test_start >= n or test_end <= test_start:
            break

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        splits.append((train_idx, test_idx))

    return splits


# ── Convenience: validate a backtest result ──────────────────

ECONOMIC_THESES = {
    "orb_0dte": (
        "Opening Range Breakout exploits institutional order flow concentration "
        "in the first 30 minutes. 0DTE options amplify directional moves with "
        "defined risk. Edge: momentum persistence after opening range resolution."
    ),
    "credit_spread": (
        "High-IVR credit spreads exploit volatility mean reversion. When IV is "
        "elevated, options are overpriced relative to realized vol. Selling "
        "premium captures the IV-RV spread. Edge: systematic theta collection."
    ),
    "earnings_spread": (
        "Pre-earnings directional spreads on stocks with analyst estimate "
        "revisions exploit informed flow ahead of announcements. Low-IVR entry "
        "ensures cheap options. Edge: information asymmetry + IV expansion."
    ),
    "straddle": (
        "Bollinger squeeze straddles profit from volatility expansion after "
        "consolidation. However, theta decay during low-vol periods creates "
        "negative carry. Edge questionable — requires precise timing."
    ),
}


def validate_backtest(strategy: str, sharpe: float, n_trades: int,
                      win_rate: float, n_trials: int = 1,
                      has_oos: bool = False) -> StrategyValidation:
    """Run full validation pipeline on a backtest result."""
    sv = StrategyValidation(strategy_name=strategy)

    sv.economic_thesis = ECONOMIC_THESES.get(strategy, "")
    sv.dsr = deflated_sharpe_ratio(sharpe, n_trials, n_trades)
    sv.sample_check = sample_size_check(n_trades, win_rate / 100)
    sv.pbo = prob_backtest_overfitting(n_trials)
    sv.has_out_of_sample = has_oos

    sv.compute_confidence()
    return sv


def print_validation_report(sv: StrategyValidation):
    """Print formatted validation report."""
    print(f"\n{'=' * 60}")
    print(f"  策略验证报告: {sv.strategy_name}")
    print(f"{'=' * 60}")

    print(f"\n  Layer 1 — 经济学假设:")
    if sv.economic_thesis:
        for line in _wrap(sv.economic_thesis, 54):
            print(f"    {line}")
    else:
        print(f"    [未记录]")

    if sv.dsr:
        d = sv.dsr
        sig = "YES" if d["is_significant"] else "NO"
        print(f"\n  Layer 2 — Deflated Sharpe Ratio:")
        print(f"    Observed Sharpe:    {d['observed_sharpe']}")
        print(f"    Noise floor:        {d['expected_max_noise_sharpe']} "
              f"({d['n_trials']} trials)")
        print(f"    DSR:                {d['deflated_sharpe']}")
        print(f"    p-value:            {d['p_value']}")
        print(f"    Significant:        {sig}")

    if sv.sample_check:
        sc = sv.sample_check
        print(f"\n  Layer 3 — 样本量检验:")
        print(f"    实际交易:           {sc['n_trades']} 笔")
        print(f"    最低要求:           {sc['min_required']} 笔")
        print(f"    充分度:             {sc['ratio']:.1%} ({sc['verdict']})")

    print(f"\n  Layer 4 — 过拟合概率 (PBO):")
    print(f"    PBO:                {sv.pbo:.1%}")

    print(f"\n  Layer 5 — 样本外验证:")
    print(f"    已完成:             {'是' if sv.has_out_of_sample else '否'}")

    print(f"\n  {'─' * 56}")
    print(f"  综合可信度:           {sv.confidence_score}/100")
    print(f"  结论:                 {sv.verdict}")
    print(f"{'=' * 60}")


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return lines
