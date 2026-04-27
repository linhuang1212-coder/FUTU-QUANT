from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from options.pricer import compute_ivr, compute_ivp, bb_width
from utils.logger import setup_logger

logger = setup_logger("options.scanner")


class VolatilityScanner:
    """Scan for options trading opportunities based on IV and volatility metrics."""

    def __init__(self, quote_ctx):
        self._ctx = quote_ctx

    def scan_ivr(self, symbols: list[str], min_ivr: float = 60.0,
                 hist_days: int = 252) -> list[dict]:
        """Find symbols with IVR above threshold (for credit spread candidates)."""
        from futu import RET_OK, KLType
        results = []
        for sym in symbols:
            import time; time.sleep(0.5)
            ret, data, _ = self._ctx.request_history_kline(
                sym, ktype=KLType.K_DAY, max_count=hist_days
            )
            if ret != RET_OK or data is None or len(data) < 60:
                continue
            closes = data["close"].values
            rets = np.diff(np.log(closes))
            hist_vols = []
            for i in range(20, len(rets)):
                window_vol = float(np.std(rets[i-20:i]) * np.sqrt(252))
                hist_vols.append(window_vol)
            if not hist_vols:
                continue
            current_vol = hist_vols[-1]
            ivr = compute_ivr(current_vol, hist_vols)
            ivp = compute_ivp(current_vol, hist_vols)
            if ivr >= min_ivr:
                results.append({
                    "symbol": sym,
                    "ivr": ivr,
                    "ivp": ivp,
                    "current_vol": round(current_vol, 4),
                    "vol_min": round(min(hist_vols), 4),
                    "vol_max": round(max(hist_vols), 4),
                })
                logger.info(f"[IVR HIT] {sym}: IVR={ivr:.1f} IVP={ivp:.1f} vol={current_vol:.2%}")
        return sorted(results, key=lambda x: x["ivr"], reverse=True)

    def scan_bb_squeeze(self, symbols: list[str], lookback: int = 126,
                        percentile: float = 5.0) -> list[dict]:
        """Find symbols with Bollinger Band Width at historical lows (squeeze)."""
        from futu import RET_OK, KLType
        results = []
        for sym in symbols:
            import time; time.sleep(0.5)
            ret, data, _ = self._ctx.request_history_kline(
                sym, ktype=KLType.K_DAY, max_count=lookback + 30
            )
            if ret != RET_OK or data is None or len(data) < lookback:
                continue
            closes = data["close"].values
            widths = bb_width(closes)
            valid = widths[~np.isnan(widths)]
            if len(valid) < lookback // 2:
                continue
            current_w = valid[-1]
            pct = float(np.percentile(valid[-lookback:], percentile))
            if current_w <= pct:
                results.append({
                    "symbol": sym,
                    "bb_width": round(float(current_w), 4),
                    "threshold": round(pct, 4),
                    "percentile_rank": round(float(np.mean(valid[-lookback:] > current_w) * 100), 1),
                })
                logger.info(f"[SQUEEZE] {sym}: BBW={current_w:.4f} <= {pct:.4f} "
                            f"(rank={results[-1]['percentile_rank']:.0f}%)")
        return results
