"""
Numba JIT-compiled hot-path functions for the backtesting pipeline.

Provides vectorised / batch replacements for:
  1. Black-Scholes pricing  (bs_price_scalar / bs_price_vec)
  2. IV-Rank computation    (precompute_ivr_fast)
  3. Credit-spread sim      (sim_spread_batch)

All @numba.njit(cache=True) — first call compiles, subsequent loads from cache.
"""
from __future__ import annotations

import math
import time

import numba
import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _norm_cdf(x: float) -> float:
    """CDF of the standard normal, numba-compatible."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@numba.njit(cache=True)
def _rolling_std_fast(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation via Welford's incremental algorithm.

    Returns an array of length len(arr) with NaN where the window is
    not yet full.  Values correspond to the std of
    arr[i-window+1 : i+1] (population std, matching np.std default).
    """
    n = len(arr)
    out = np.empty(n, dtype=np.float64)

    mean = 0.0
    m2 = 0.0

    for i in range(n):
        if i < window:
            # Build-up phase — accumulate but don't emit until window full
            delta = arr[i] - mean
            mean += delta / (i + 1)
            delta2 = arr[i] - mean
            m2 += delta * delta2

            if i == window - 1:
                out[i] = math.sqrt(m2 / window)
            else:
                out[i] = np.nan
        else:
            old = arr[i - window]
            new = arr[i]

            old_mean = mean
            mean += (new - old) / window
            m2 += (new - mean + old - old_mean) * (new - old)

            if m2 < 0.0:
                m2 = 0.0
            out[i] = math.sqrt(m2 / window)

    return out


# ═══════════════════════════════════════════════════════════════════
#  1. Black-Scholes — scalar & vectorised
# ═══════════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def bs_price_scalar(S: float, K: float, T: float, r: float,
                    sigma: float, is_call: bool) -> float:
    """Black-Scholes price for a single option (njit-compiled).

    Parameters
    ----------
    is_call : True for CALL, False for PUT
    """
    if T <= 0.0 or sigma <= 0.0:
        if is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


@numba.njit(cache=True)
def bs_price_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray,
                 r: np.ndarray, sigma: np.ndarray,
                 is_call: np.ndarray) -> np.ndarray:
    """Vectorised Black-Scholes over aligned 1-D arrays.

    All inputs must be float64 arrays of the same length.
    ``is_call`` is a boolean (or int8) array — True / 1 for CALL.
    """
    n = len(S)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = bs_price_scalar(S[i], K[i], T[i], r[i],
                                 sigma[i], bool(is_call[i]))
    return out


# ═══════════════════════════════════════════════════════════════════
#  2. Vectorised IV-Rank
# ═══════════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def precompute_ivr_fast(closes: np.ndarray) -> np.ndarray:
    """Compute IV-Rank array from a close-price series.

    Mirrors the Python ``_precompute_ivr`` but replaces the O(n*252)
    double loop + np.std slices with Welford rolling std.

    Returns array same length as *closes*, with NaN before index 252.
    """
    n = len(closes)
    ivr = np.full(n, np.nan, dtype=np.float64)

    if n < 253:
        return ivr

    # Log returns
    rets = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        rets[i] = math.log(closes[i + 1] / closes[i])

    sqrt252 = math.sqrt(252.0)

    # Rolling 20-day std of returns (annualised)
    vol20 = _rolling_std_fast(rets, 20)
    for i in range(len(vol20)):
        vol20[i] *= sqrt252

    # For each bar from 252 onward, compute IVR
    for i in range(252, n):
        # vol20 is indexed on *rets* (length n-1), so bar i of closes
        # corresponds to rets index i-1.  The 20-day vol ending at bar i
        # uses rets[i-20 : i], which is vol20[i-1].
        ret_idx = i - 1
        if ret_idx < 19:
            continue
        current_vol = vol20[ret_idx]
        if current_vol != current_vol:  # NaN check
            continue

        # Scan historical vols every 5 bars over the past year
        vmin = 1e30
        vmax = -1e30
        count = 0

        start_j = i - 252
        if start_j < 20:
            start_j = 20

        j = start_j
        while j < i:
            rj = j - 1  # corresponding rets index
            if rj >= 19:
                wv = vol20[rj]
                if wv == wv:  # not NaN
                    if wv < vmin:
                        vmin = wv
                    if wv > vmax:
                        vmax = wv
                    count += 1
            j += 5

        if count > 0 and (vmax - vmin) > 0.001:
            ivr[i] = (current_vol - vmin) / (vmax - vmin) * 100.0
            if ivr[i] < 0.0:
                ivr[i] = 0.0
            elif ivr[i] > 100.0:
                ivr[i] = 100.0
        elif count > 0:
            ivr[i] = 50.0

    return ivr


# ═══════════════════════════════════════════════════════════════════
#  3. Batch credit-spread simulation
# ═══════════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def sim_spread_batch(
    closes: np.ndarray,
    ivr_arr: np.ndarray,
    spread_width: float,
    target_delta: float,
    max_hold: int,
    tp_pct: float,
    sl_pct: float,
    r: float,
    min_ivr: float,
    commission: float,
    spread_pct: float,
) -> np.ndarray:
    """Simulate bull-put credit spreads over the full close-price series.

    Walks *closes* from index 252 onward.  Enters a spread when
    ``ivr >= min_ivr`` and the spot is positive.  Each trade holds up to
    *max_hold* days, exiting early on take-profit or stop-loss.

    Transaction costs are modelled as:
        cost_per_leg = commission + avg_option_price * spread_pct * 100

    Returns a 1-D array of per-trade PnLs.
    """
    n = len(closes)
    # Pre-allocate generous buffer (at most n trades)
    pnl_buf = np.empty(n, dtype=np.float64)
    trade_count = 0

    i = 252
    while i < n:
        # --- entry filter ---
        if i >= len(ivr_arr):
            i += 1
            continue
        ivr_val = ivr_arr[i]
        if ivr_val != ivr_val:  # NaN
            i += 1
            continue
        if ivr_val < min_ivr:
            i += 1
            continue

        spot = closes[i]
        if spot <= 0.0:
            i += 1
            continue

        # Synthesise IV from a 20-day realised vol proxy
        # (rets already embedded — use a simple backward std)
        vol_sum = 0.0
        vol_sum2 = 0.0
        vcount = 0
        for k in range(max(1, i - 20), i):
            lr = math.log(closes[k] / closes[k - 1])
            vol_sum += lr
            vol_sum2 += lr * lr
            vcount += 1

        if vcount < 2:
            i += 1
            continue

        rv_mean = vol_sum / vcount
        rv_var = vol_sum2 / vcount - rv_mean * rv_mean
        if rv_var < 0.0:
            rv_var = 0.0
        iv = math.sqrt(rv_var) * math.sqrt(252.0) * 1.2  # IV ≈ RV * 1.2 smile uplift

        if iv <= 0.0:
            i += 1
            continue

        T = max_hold / 252.0

        # Bull put: sell OTM put, buy further OTM put
        short_strike = round(spot * (1.0 - target_delta * 0.5))
        long_strike = short_strike - spread_width
        if short_strike <= 0.0 or long_strike <= 0.0:
            i += 1
            continue

        short_p = bs_price_scalar(spot, short_strike, T, r, iv, False)
        long_p = bs_price_scalar(spot, long_strike, T, r, iv, False)
        credit = short_p - long_p
        if credit <= 0.05:
            i += 1
            continue

        avg_price = (short_p + long_p) / 2.0
        entry_cost = 2.0 * (commission + max(0.01, avg_price * spread_pct) * 100.0)

        max_loss = (abs(long_strike - short_strike) - credit) * 100.0
        pnl = credit * 100.0
        exit_avg = avg_price

        # --- day loop ---
        end_day = min(max_hold + 1, n - i)
        for d in range(1, end_day):
            future_spot = closes[i + d]
            T_rem = (max_hold - d) / 252.0
            if T_rem < 0.001:
                T_rem = 0.001

            short_now = bs_price_scalar(future_spot, short_strike, T_rem, r, iv, False)
            long_now = bs_price_scalar(future_spot, long_strike, T_rem, r, iv, False)
            spread_now = short_now - long_now
            cur_pnl = (credit - spread_now) * 100.0

            if cur_pnl >= credit * 100.0 * tp_pct:
                pnl = cur_pnl
                exit_avg = (short_now + long_now) / 2.0
                break
            if cur_pnl <= -max_loss * sl_pct:
                pnl = cur_pnl
                exit_avg = (short_now + long_now) / 2.0
                break
            pnl = cur_pnl
            exit_avg = (short_now + long_now) / 2.0

        exit_cost = 2.0 * (commission + max(0.01, exit_avg * spread_pct) * 100.0)
        pnl_after = pnl - entry_cost - exit_cost

        pnl_buf[trade_count] = pnl_after
        trade_count += 1

        # Skip ahead past the holding period before looking for next entry
        i += max_hold + 1

    return pnl_buf[:trade_count]


# ═══════════════════════════════════════════════════════════════════
#  Verification
# ═══════════════════════════════════════════════════════════════════

def _bs_price_reference(S, K, T, r, sigma, is_call):
    """Pure-Python reference BS price (no numba)."""
    if T <= 0 or sigma <= 0:
        if is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    cdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    if is_call:
        return S * cdf(d1) - K * math.exp(-r * T) * cdf(d2)
    return K * math.exp(-r * T) * cdf(-d2) - S * cdf(-d1)


def verify_turbo_core():
    """Warm-up + correctness / timing tests."""
    print("=" * 60)
    print("  turbo_core  —  verification suite")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. bs_price_scalar correctness
    # ----------------------------------------------------------
    print("\n[1] bs_price_scalar vs reference")

    test_cases = [
        (100.0, 100.0, 0.25, 0.05, 0.20, True),
        (100.0, 100.0, 0.25, 0.05, 0.20, False),
        (50.0, 55.0, 1.0, 0.02, 0.30, True),
        (50.0, 55.0, 1.0, 0.02, 0.30, False),
        (200.0, 180.0, 0.5, 0.04, 0.25, True),
        (100.0, 100.0, 0.0, 0.05, 0.20, True),   # edge: T=0
        (100.0, 100.0, 0.25, 0.05, 0.0, False),   # edge: sigma=0
    ]

    # Warm-up compile
    _ = bs_price_scalar(100.0, 100.0, 0.25, 0.05, 0.2, True)

    all_ok = True
    for S, K, T, r, sig, ic in test_cases:
        jit_val = bs_price_scalar(S, K, T, r, sig, ic)
        ref_val = _bs_price_reference(S, K, T, r, sig, ic)
        diff = abs(jit_val - ref_val)
        ok = diff < 1e-8
        tag = "OK" if ok else "FAIL"
        print(f"  {tag}  S={S} K={K} T={T} σ={sig} {'C' if ic else 'P'}"
              f"  jit={jit_val:.6f}  ref={ref_val:.6f}  Δ={diff:.2e}")
        if not ok:
            all_ok = False
    print(f"  -> {'ALL PASSED' if all_ok else 'SOME FAILURES'}")

    # ----------------------------------------------------------
    # 2. bs_price_vec timing
    # ----------------------------------------------------------
    print("\n[2] bs_price_vec timing (100 000 options)")
    n = 100_000
    rng = np.random.default_rng(42)
    S_arr = rng.uniform(50, 200, n)
    K_arr = rng.uniform(50, 200, n)
    T_arr = rng.uniform(0.01, 1.0, n)
    r_arr = np.full(n, 0.05)
    sig_arr = rng.uniform(0.10, 0.60, n)
    ic_arr = rng.choice(np.array([True, False]), n)

    # Warm-up
    _ = bs_price_vec(S_arr[:10], K_arr[:10], T_arr[:10],
                     r_arr[:10], sig_arr[:10], ic_arr[:10])

    t0 = time.perf_counter()
    prices = bs_price_vec(S_arr, K_arr, T_arr, r_arr, sig_arr, ic_arr)
    t_jit = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref_prices = np.array([
        _bs_price_reference(S_arr[i], K_arr[i], T_arr[i],
                            r_arr[i], sig_arr[i], bool(ic_arr[i]))
        for i in range(n)
    ])
    t_py = time.perf_counter() - t0

    max_err = float(np.max(np.abs(prices - ref_prices)))
    print(f"  JIT:  {t_jit*1000:.1f} ms")
    print(f"  Py:   {t_py*1000:.1f} ms")
    print(f"  Speedup: {t_py/t_jit:.1f}x")
    print(f"  Max error: {max_err:.2e}")

    # ----------------------------------------------------------
    # 3. precompute_ivr_fast
    # ----------------------------------------------------------
    print("\n[3] precompute_ivr_fast")
    np.random.seed(0)
    fake_close = 100.0 * np.exp(np.cumsum(np.random.randn(600) * 0.01))

    # Warm-up
    _ = precompute_ivr_fast(fake_close[:300])

    t0 = time.perf_counter()
    ivr = precompute_ivr_fast(fake_close)
    t_ivr = time.perf_counter() - t0

    valid = ivr[~np.isnan(ivr)]
    print(f"  Length: {len(ivr)}, non-NaN: {len(valid)}")
    if len(valid) > 0:
        print(f"  Range: [{valid.min():.1f}, {valid.max():.1f}]")
    print(f"  Time:  {t_ivr*1000:.2f} ms")
    print(f"  -> {'OK' if len(valid) > 0 else 'FAIL (no valid IVR)'}")

    # Reference: slow Python version for comparison
    from options.pricer import compute_ivr as _compute_ivr_py
    rets_ref = np.diff(np.log(fake_close))
    t0 = time.perf_counter()
    ivr_ref = np.full(len(fake_close), np.nan)
    for idx in range(252, len(fake_close)):
        cv = float(np.std(rets_ref[idx - 20:idx]) * np.sqrt(252))
        vols = []
        for jj in range(max(20, idx - 252), idx, 5):
            wv = float(np.std(rets_ref[max(0, jj - 20):jj]) * np.sqrt(252))
            vols.append(wv)
        if vols:
            ivr_ref[idx] = _compute_ivr_py(cv, vols)
    t_py_ivr = time.perf_counter() - t0
    print(f"  Python IVR time: {t_py_ivr*1000:.2f} ms")
    print(f"  IVR speedup: {t_py_ivr / max(t_ivr, 1e-9):.1f}x")

    # ----------------------------------------------------------
    # 4. sim_spread_batch
    # ----------------------------------------------------------
    print("\n[4] sim_spread_batch")
    np.random.seed(1)
    closes = 100.0 * np.exp(np.cumsum(np.random.randn(1000) * 0.01))
    ivr_arr = precompute_ivr_fast(closes)

    # Warm-up
    _ = sim_spread_batch(closes[:300], ivr_arr[:300],
                         5.0, 0.30, 30, 0.50, 0.80,
                         0.05, 30.0, 0.65, 0.02)

    t0 = time.perf_counter()
    pnls = sim_spread_batch(closes, ivr_arr,
                            5.0, 0.30, 30, 0.50, 0.80,
                            0.05, 30.0, 0.65, 0.02)
    t_sim = time.perf_counter() - t0

    print(f"  Trades: {len(pnls)}")
    if len(pnls) > 0:
        print(f"  Avg PnL: ${np.mean(pnls):.2f}")
        print(f"  Total PnL: ${np.sum(pnls):.2f}")
    print(f"  Time: {t_sim*1000:.2f} ms")
    print(f"  -> {'OK' if len(pnls) > 0 else 'WARN (no trades — check IVR threshold)'}")

    # ----------------------------------------------------------
    # 5. _rolling_std_fast sanity
    # ----------------------------------------------------------
    print("\n[5] _rolling_std_fast sanity check")
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    window = 3
    rs = _rolling_std_fast(test_data, window)
    ref_stds = np.array([
        np.std(test_data[max(0, i - window + 1):i + 1])
        if i >= window - 1 else np.nan
        for i in range(len(test_data))
    ])
    ok = True
    for i in range(len(test_data)):
        if np.isnan(rs[i]) and np.isnan(ref_stds[i]):
            continue
        if abs(rs[i] - ref_stds[i]) > 1e-6:
            print(f"  MISMATCH at i={i}: fast={rs[i]:.6f} ref={ref_stds[i]:.6f}")
            ok = False
    print(f"  -> {'OK' if ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print("  Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    verify_turbo_core()
