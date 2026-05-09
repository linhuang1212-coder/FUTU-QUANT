from __future__ import annotations

import pandas as pd

from factor.processor import cross_sectional_rank
from utils.logger import setup_logger

logger = setup_logger("factor.scorer")


class FactorScorer:
    """Combine multiple factors into a single composite score."""

    def __init__(self, factors: dict[str, pd.DataFrame],
                 weights: dict[str, float] | None = None):
        self.factors = factors
        self.weights = weights or {k: 1.0 / len(factors) for k in factors}

    def score_equal(self) -> pd.DataFrame:
        """Equal-weight composite using rank-standardized factors."""
        ranked = {name: cross_sectional_rank(panel)
                  for name, panel in self.factors.items()}

        keys = list(ranked.keys())
        common_idx = ranked[keys[0]].index
        common_cols = ranked[keys[0]].columns
        for panel in ranked.values():
            common_idx = common_idx.intersection(panel.index)
            common_cols = common_cols.intersection(panel.columns)

        composite = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
        w_total = sum(self.weights.get(n, 0) for n in ranked)
        if w_total == 0:
            return composite

        for name, panel in ranked.items():
            w = self.weights.get(name, 0) / w_total
            composite += panel.loc[common_idx, common_cols] * w
        return composite

    def score_ic_weighted(self, ic_table: pd.DataFrame,
                          min_ic_ir: float = 0.0) -> pd.DataFrame:
        """IC_IR-weighted composite. Flips negative-IC factors automatically."""
        valid_factors = {}
        ic_weights = {}

        for name in self.factors:
            if name not in ic_table.index:
                continue
            ic_ir = ic_table.loc[name, "IC_IR"]
            if abs(ic_ir) <= min_ic_ir:
                continue
            valid_factors[name] = self.factors[name]
            ic_weights[name] = abs(ic_ir)
            if ic_ir < 0:
                valid_factors[name] = -valid_factors[name]

        if not valid_factors:
            logger.warning("No factors pass IC_IR threshold, falling back to equal weight")
            return self.score_equal()

        sub = FactorScorer(valid_factors, ic_weights)
        logger.info(f"IC-weighted scoring: {list(valid_factors.keys())}")
        return sub.score_equal()

    def rank_symbols(self, date=None) -> list[tuple[str, float]]:
        """Rank symbols by composite score for a given date (default: latest)."""
        composite = self.score_equal()
        if composite.empty:
            return []

        if date is None:
            date = composite.index[-1]
        elif date not in composite.index:
            idx = composite.index.get_indexer([date], method="ffill")
            if idx[0] >= 0:
                date = composite.index[idx[0]]
            else:
                return []

        scores = composite.loc[date].dropna().sort_values(ascending=False)
        return list(zip(scores.index, scores.values))
