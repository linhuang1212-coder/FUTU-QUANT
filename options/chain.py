from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("options.chain")


class OptionChainFetcher:
    """Wraps Futu OpenAPI for option chain queries with caching."""

    def __init__(self, quote_ctx):
        self._ctx = quote_ctx
        self._cache: dict[str, tuple[str, object]] = {}  # key -> (date_str, data)

    def _cache_key(self, *args) -> str:
        return "|".join(str(a) for a in args)

    def _get_cached(self, key: str):
        today = datetime.now().strftime("%Y-%m-%d")
        if key in self._cache and self._cache[key][0] == today:
            return self._cache[key][1]
        return None

    def _set_cache(self, key: str, data):
        today = datetime.now().strftime("%Y-%m-%d")
        self._cache[key] = (today, data)

    def get_expiry_dates(self, underlying: str) -> list[dict]:
        from futu import RET_OK
        key = self._cache_key("expiry", underlying)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        time.sleep(0.3)
        ret, data = self._ctx.get_option_expiration_date(code=underlying)
        if ret != RET_OK or data is None:
            logger.error(f"Failed to get expiry dates for {underlying}: {data}")
            return []
        result = data.to_dict("records")
        self._set_cache(key, result)
        return result

    def get_chain(self, underlying: str, start: str, end: str,
                  option_type: str = "ALL", cond_type: str = "ALL"):
        """Fetch option chain. Returns DataFrame or None."""
        from futu import RET_OK, OptionType, OptionCondType
        type_map = {"ALL": OptionType.ALL, "CALL": OptionType.CALL, "PUT": OptionType.PUT}
        cond_map = {"ALL": OptionCondType.ALL, "WITHIN": OptionCondType.WITHIN,
                     "OUTSIDE": OptionCondType.OUTSIDE}

        key = self._cache_key("chain", underlying, start, end, option_type, cond_type)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        time.sleep(0.3)
        ret, data = self._ctx.get_option_chain(
            code=underlying, start=start, end=end,
            option_type=type_map.get(option_type, OptionType.ALL),
            option_cond_type=cond_map.get(cond_type, OptionCondType.ALL),
        )
        if ret != RET_OK or data is None:
            logger.error(f"Failed to get option chain for {underlying}: {data}")
            return None
        self._set_cache(key, data)
        return data

    def get_option_quote(self, option_codes: list[str]) -> Optional[dict]:
        """Get snapshot quotes for option codes. Returns {code: {last_price, bid, ask, ...}}."""
        from futu import RET_OK
        if not option_codes:
            return None
        time.sleep(0.3)
        ret, data = self._ctx.get_market_snapshot(option_codes)
        if ret != RET_OK or data is None:
            logger.error(f"Failed to get option quotes: {data}")
            return None
        result = {}
        for _, row in data.iterrows():
            result[row["code"]] = {
                "last_price": float(row.get("last_price", 0)),
                "bid_price": float(row.get("bid_price", 0)),
                "ask_price": float(row.get("ask_price", 0)),
                "volume": int(row.get("volume", 0)),
                "open_interest": int(row.get("open_interest", 0)),
            }
        return result

    def find_atm_option(self, underlying: str, expiry: str, option_type: str = "CALL"):
        """Find the ATM option closest to current price."""
        from futu import RET_OK
        time.sleep(0.3)
        ret, snap = self._ctx.get_market_snapshot([underlying])
        if ret != RET_OK or snap is None or len(snap) == 0:
            logger.error(f"Cannot get price for {underlying}")
            return None
        current_price = float(snap.iloc[0]["last_price"])

        chain = self.get_chain(underlying, expiry, expiry, option_type=option_type)
        if chain is None or len(chain) == 0:
            return None

        chain = chain.copy()
        chain["strike_diff"] = abs(chain["strike_price"] - current_price)
        best = chain.sort_values("strike_diff").iloc[0]
        return {
            "code": best["code"],
            "strike": float(best["strike_price"]),
            "option_type": option_type,
            "expiry": expiry,
            "underlying_price": current_price,
        }

    def find_by_delta(self, underlying: str, expiry: str, option_type: str, target_delta: float):
        """Find option closest to target delta using Black-Scholes."""
        from options.pricer import greeks as bs_greeks
        chain = self.get_chain(underlying, expiry, expiry, option_type=option_type)
        if chain is None or len(chain) == 0:
            return None

        from futu import RET_OK
        time.sleep(0.3)
        ret, snap = self._ctx.get_market_snapshot([underlying])
        if ret != RET_OK or snap is None:
            return None
        spot = float(snap.iloc[0]["last_price"])

        # Days to expiry for BS calculation
        from datetime import datetime
        try:
            exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
            dte = max((exp_dt - datetime.now()).days, 1)
        except ValueError:
            dte = 21
        T = dte / 365.0
        r = 0.05
        sigma = 0.30  # reasonable IV estimate

        best_code, best_diff = None, float("inf")
        best_strike = 0.0
        best_delta = 0.0
        for _, row in chain.iterrows():
            strike = float(row["strike_price"])
            g = bs_greeks(spot, strike, T, r, sigma, option_type)
            delta = abs(g["delta"])
            diff = abs(delta - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_code = row["code"]
                best_strike = strike
                best_delta = delta

        if best_code is None:
            return None
        logger.info(f"[DELTA] {underlying} {option_type} target={target_delta:.2f} "
                    f"-> strike={best_strike:.0f} (delta={best_delta:.3f}, "
                    f"spot={spot:.2f}, DTE={dte})")
        return {"code": best_code, "strike": best_strike,
                "option_type": option_type, "expiry": expiry,
                "delta": best_delta}
