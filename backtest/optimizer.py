import itertools
import pandas as pd
from typing import Type
from backtest.backtester import Backtester
from backtest.report import BacktestReport
from strategy.base import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("optimizer")


class ParameterOptimizer:
    """Grid search parameter optimizer with train/validation split."""

    def __init__(self, initial_capital: float = 3000, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def grid_search(
        self,
        strategy_class: Type[BaseStrategy],
        symbol: str,
        data: pd.DataFrame,
        param_grid: dict,
        sort_by: str = "sharpe_ratio",
        ascending: bool = False,
        min_trades: int = 2,
    ) -> pd.DataFrame:
        """
        Run grid search over all parameter combinations.

        param_grid example: {
            "fast_ma_period": [3, 5, 8],
            "slow_ma_period": [10, 15, 20],
            "rsi_period": [7, 10, 14],
        }

        Returns DataFrame with one row per parameter combo, sorted by sort_by.
        """
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        results = []
        total = len(combinations)

        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))

            try:
                strategy = strategy_class(params=params)
                bt = Backtester(self.initial_capital, self.commission_pct, self.slippage_pct)
                result = bt.run(strategy, symbol, data)
                report = BacktestReport(result)
                summary = report.summary()

                row = {**params, **summary}
                results.append(row)
            except Exception as e:
                logger.warning(f"Combo {idx+1}/{total} failed: {params} -> {e}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Filter by minimum trades
        if min_trades > 0 and "total_trades" in df.columns:
            df = df[df["total_trades"] >= min_trades]

        # Sort
        if sort_by in df.columns:
            # Handle inf values for sorting
            df_sort = df.copy()
            df_sort[sort_by] = df_sort[sort_by].replace([float("inf"), float("-inf")], float("nan"))
            df = df.loc[df_sort.sort_values(sort_by, ascending=ascending, na_position="last").index]

        return df.reset_index(drop=True)

    def walk_forward(
        self,
        strategy_class: Type[BaseStrategy],
        symbol: str,
        data: pd.DataFrame,
        param_grid: dict,
        train_pct: float = 0.7,
        sort_by: str = "sharpe_ratio",
        min_trades: int = 2,
    ) -> dict:
        """
        Split data into train/validation, optimize on train, validate on test.
        Returns dict with train_results, validation_results, and best_params.
        Helps detect overfitting.
        """
        split_idx = int(len(data) * train_pct)
        train_data = data.iloc[:split_idx].reset_index(drop=True)
        val_data = data.iloc[split_idx:].reset_index(drop=True)

        # Optimize on training set
        train_results = self.grid_search(
            strategy_class, symbol, train_data, param_grid, sort_by=sort_by, min_trades=min_trades
        )

        if train_results.empty:
            return {
                "train_results": train_results,
                "validation_results": pd.DataFrame(),
                "best_params": {},
                "overfit_score": None,
            }

        # Get parameter columns (not metric columns)
        param_keys = list(param_grid.keys())

        # Validate top N on validation set
        top_n = min(10, len(train_results))
        val_results = []

        for i in range(top_n):
            row = train_results.iloc[i]
            params = {k: row[k] for k in param_keys}

            # Convert numpy types to python types for strategy constructor
            clean_params = {}
            for k, v in params.items():
                if hasattr(v, "item"):
                    clean_params[k] = v.item()
                else:
                    clean_params[k] = v

            try:
                strategy = strategy_class(params=clean_params)
                bt = Backtester(self.initial_capital, self.commission_pct, self.slippage_pct)
                result = bt.run(strategy, symbol, val_data)
                report = BacktestReport(result)
                summary = report.summary()

                val_row = {**clean_params, "train_rank": i + 1, **summary}
                val_results.append(val_row)
            except Exception as e:
                logger.warning(f"Validation failed for params {params}: {e}")

        val_df = pd.DataFrame(val_results) if val_results else pd.DataFrame()

        # Best params: the one that ranks well in BOTH train and validation
        best_params = {}
        overfit_score = None
        if not val_df.empty and sort_by in val_df.columns:
            val_sorted = val_df.copy()
            val_sorted[sort_by] = val_sorted[sort_by].replace([float("inf"), float("-inf")], float("nan"))
            val_sorted = val_sorted.sort_values(sort_by, ascending=False, na_position="last")
            best_row = val_sorted.iloc[0]
            best_params = {k: best_row[k] for k in param_keys}

            # Clean numpy types
            for k, v in best_params.items():
                if hasattr(v, "item"):
                    best_params[k] = v.item()

            # Overfit score: how much worse validation is vs training
            # Find this param set's train performance
            train_perf = train_results.iloc[0].get(sort_by, 0)
            val_perf = best_row.get(sort_by, 0)
            if train_perf and train_perf != 0 and not pd.isna(train_perf) and not pd.isna(val_perf):
                overfit_score = (
                    round((1 - val_perf / train_perf) * 100, 2) if train_perf != float("inf") else None
                )

        return {
            "train_results": train_results,
            "validation_results": val_df,
            "best_params": best_params,
            "overfit_score": overfit_score,
        }

    def multi_symbol_scan(
        self,
        strategy_class: Type[BaseStrategy],
        symbols_data: dict,
        param_grid: dict,
        sort_by: str = "sharpe_ratio",
        min_trades: int = 2,
    ) -> pd.DataFrame:
        """
        Run grid search across multiple symbols.
        symbols_data: {"US.TQQQ": df1, "US.SOXL": df2, ...}
        Returns combined DataFrame with a 'symbol' column.
        """
        all_results = []

        for symbol, data in symbols_data.items():
            logger.info(f"Scanning {symbol}...")
            results = self.grid_search(
                strategy_class, symbol, data, param_grid, sort_by=sort_by, min_trades=min_trades
            )
            if not results.empty:
                results.insert(0, "symbol", symbol)
                all_results.append(results)

        if not all_results:
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)

        if sort_by in combined.columns:
            combined_sort = combined.copy()
            combined_sort[sort_by] = combined_sort[sort_by].replace([float("inf"), float("-inf")], float("nan"))
            combined = combined.loc[
                combined_sort.sort_values(sort_by, ascending=False, na_position="last").index
            ]

        return combined.reset_index(drop=True)

    def save_results(self, results: pd.DataFrame, filepath: str) -> None:
        """Save results to CSV."""
        from pathlib import Path

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
