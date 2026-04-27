"""backtest/optimizer.py  —  并行加速版

改动要点：
  - grid_search：使用 ProcessPoolExecutor 并行跑所有参数组合
  - walk_forward：train 阶段并行，val 阶段并行
  - multi_symbol_scan：每个 symbol 并行
  - Windows 兼容：所有并行任务用模块级函数（不用 lambda / 嵌套函数）
  - n_jobs=-1 自动用全部逻辑核心；n_jobs=1 退化为串行（调试用）
"""

import itertools
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Type
from backtest.backtester import Backtester
from backtest.report import BacktestReport
from strategy.base import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("optimizer")

# ── 模块级 worker 函数（Windows multiprocessing 要求顶层可 pickle）──────────

def _run_combo(args: tuple) -> dict | None:
    """单个参数组合回测 worker。"""
    strategy_class, params, symbol, data, initial_capital, commission_pct, slippage_pct = args
    try:
        strategy = strategy_class(params=params)
        bt = Backtester(initial_capital, commission_pct, slippage_pct)
        result = bt.run(strategy, symbol, data)
        from backtest.report import BacktestReport
        report = BacktestReport(result)
        summary = report.summary()
        return {**params, **summary}
    except Exception as e:
        return None


def _run_val_combo(args: tuple) -> dict | None:
    """Walk-forward 验证阶段 worker。"""
    strategy_class, params, train_rank, symbol, val_data, initial_capital, commission_pct, slippage_pct, sort_by = args
    try:
        strategy = strategy_class(params=params)
        bt = Backtester(initial_capital, commission_pct, slippage_pct)
        result = bt.run(strategy, symbol, val_data)
        from backtest.report import BacktestReport
        report = BacktestReport(result)
        summary = report.summary()
        return {**params, "train_rank": train_rank, **summary}
    except Exception as e:
        return None


# ── ParameterOptimizer ────────────────────────────────────────────────────────

class ParameterOptimizer:
    """Grid search parameter optimizer with train/validation split.

    Args:
        n_jobs: 并行进程数。-1 = 全部逻辑核心，1 = 串行（调试）。
    """

    def __init__(
        self,
        initial_capital: float = 3000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        n_jobs: int = -1,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.n_jobs = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)

    # ── grid_search ───────────────────────────────────────────────────────────

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
        """并行网格搜索。"""
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*param_grid.values()))
        total = len(combinations)

        # 构造 worker 参数列表
        task_args = [
            (
                strategy_class,
                dict(zip(keys, combo)),
                symbol,
                data,
                self.initial_capital,
                self.commission_pct,
                self.slippage_pct,
            )
            for combo in combinations
        ]

        results = []
        if self.n_jobs == 1:
            # 串行模式（调试用）
            for i, args in enumerate(task_args):
                row = _run_combo(args)
                if row is not None:
                    results.append(row)
                if (i + 1) % 50 == 0:
                    logger.info(f"  grid_search: {i+1}/{total}")
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                futures = {pool.submit(_run_combo, a): i for i, a in enumerate(task_args)}
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    row = fut.result()
                    if row is not None:
                        results.append(row)
                    if done % 50 == 0:
                        logger.info(f"  grid_search: {done}/{total}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        if min_trades > 0 and "total_trades" in df.columns:
            df = df[df["total_trades"] >= min_trades]

        if sort_by in df.columns:
            df_sort = df.copy()
            df_sort[sort_by] = df_sort[sort_by].replace([float("inf"), float("-inf")], float("nan"))
            df = df.loc[df_sort.sort_values(sort_by, ascending=ascending, na_position="last").index]

        return df.reset_index(drop=True)

    # ── walk_forward ──────────────────────────────────────────────────────────

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
        """Train 阶段 + 验证阶段均并行。"""
        split_idx = int(len(data) * train_pct)
        train_data = data.iloc[:split_idx].reset_index(drop=True)
        val_data = data.iloc[split_idx:].reset_index(drop=True)

        # 并行跑 train
        train_results = self.grid_search(
            strategy_class, symbol, train_data, param_grid,
            sort_by=sort_by, min_trades=min_trades
        )

        if train_results.empty:
            return {
                "train_results": train_results,
                "validation_results": pd.DataFrame(),
                "best_params": {},
                "overfit_score": None,
            }

        param_keys = list(param_grid.keys())
        top_n = min(10, len(train_results))

        # 并行跑 val
        val_task_args = []
        for i in range(top_n):
            row = train_results.iloc[i]
            params = {k: row[k] for k in param_keys}
            clean_params = {k: v.item() if hasattr(v, "item") else v for k, v in params.items()}
            val_task_args.append((
                strategy_class, clean_params, i + 1, symbol, val_data,
                self.initial_capital, self.commission_pct, self.slippage_pct, sort_by,
            ))

        val_results = []
        if self.n_jobs == 1:
            for args in val_task_args:
                row = _run_val_combo(args)
                if row is not None:
                    val_results.append(row)
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                futures = [pool.submit(_run_val_combo, a) for a in val_task_args]
                for fut in as_completed(futures):
                    row = fut.result()
                    if row is not None:
                        val_results.append(row)

        val_df = pd.DataFrame(val_results) if val_results else pd.DataFrame()

        best_params = {}
        overfit_score = None
        if not val_df.empty and sort_by in val_df.columns:
            val_sorted = val_df.copy()
            val_sorted[sort_by] = val_sorted[sort_by].replace([float("inf"), float("-inf")], float("nan"))
            val_sorted = val_sorted.sort_values(sort_by, ascending=False, na_position="last")
            best_row = val_sorted.iloc[0]
            best_params = {k: best_row[k] for k in param_keys}
            best_params = {k: v.item() if hasattr(v, "item") else v for k, v in best_params.items()}

            train_perf = train_results.iloc[0].get(sort_by, 0)
            val_perf = best_row.get(sort_by, 0)
            if train_perf and train_perf != 0 and not pd.isna(train_perf) and not pd.isna(val_perf):
                overfit_score = (
                    round((1 - val_perf / train_perf) * 100, 2)
                    if train_perf != float("inf") else None
                )

        return {
            "train_results": train_results,
            "validation_results": val_df,
            "best_params": best_params,
            "overfit_score": overfit_score,
        }

    # ── multi_symbol_scan ─────────────────────────────────────────────────────

    def multi_symbol_scan(
        self,
        strategy_class: Type[BaseStrategy],
        symbols_data: dict,
        param_grid: dict,
        sort_by: str = "sharpe_ratio",
        min_trades: int = 2,
    ) -> pd.DataFrame:
        """每个 symbol 并行（symbol 级粗粒度并行）。"""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        def _scan_symbol(sym):
            results = self.grid_search(
                strategy_class, sym, symbols_data[sym],
                param_grid, sort_by=sort_by, min_trades=min_trades
            )
            if not results.empty:
                results.insert(0, "symbol", sym)
            return results

        all_results = []
        # symbol 数量少，直接用线程池避免 pickle DataFrame 开销
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(symbols_data))) as pool:
            futures = {pool.submit(_scan_symbol, sym): sym for sym in symbols_data}
            for fut in as_completed(futures):
                r = fut.result()
                if not r.empty:
                    all_results.append(r)

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
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
