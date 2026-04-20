"""Synthesize intraday (5-min) bars from daily OHLCV data.

Generates a realistic 78-bar (6.5h) price path per trading day using
the daily open/high/low/close as constraints. This allows backtesting
intraday strategies over 10+ years of daily data instead of the 60 days
available from free 5-min data sources.

The synthetic path is NOT random — it follows a structured pattern:
  1. Open at daily open
  2. Move toward one extreme (high or low) based on daily direction
  3. Range-bound middle section
  4. Move toward close
  5. Close at daily close

Volume is distributed using a U-shaped curve (heavy at open and close).

Usage:
    from data.synthesizer import synthesize_intraday
    df_5min = synthesize_intraday(df_daily)
"""

import numpy as np
import pandas as pd
from typing import Optional

BARS_PER_DAY = 78  # 6.5 hours * 12 bars/hour


def _generate_price_path(o: float, h: float, l: float, c: float,
                         n_bars: int = BARS_PER_DAY) -> np.ndarray:
    """Generate a plausible intraday price path from OHLC constraints.

    Returns array of shape (n_bars,) representing close prices for each bar.
    """
    if h <= l or h < max(o, c) or l > min(o, c):
        return np.linspace(o, c, n_bars)

    up_day = c >= o

    # Phase allocation (bars): open_drive, extreme1, mid_range, extreme2, close_drive
    phase_1 = max(3, n_bars // 6)      # initial direction (first ~30 min)
    phase_5 = max(6, n_bars // 5)      # close drive (last ~30 min)
    phase_mid = n_bars - phase_1 - phase_5
    phase_2 = phase_mid // 2
    phase_3 = phase_mid - phase_2

    path = np.empty(n_bars)
    idx = 0

    if up_day:
        # Phase 1: open -> dip toward low (morning weakness)
        dip_target = l + (o - l) * 0.3
        path[idx:idx + phase_1] = np.linspace(o, dip_target, phase_1)
        idx += phase_1

        # Phase 2: recover from dip toward high
        path[idx:idx + phase_2] = np.linspace(dip_target, h * 0.98, phase_2)
        idx += phase_2

        # Phase 3: range near high
        mid_level = (h + c) / 2
        noise_amp = (h - l) * 0.05
        base = np.linspace(h * 0.98, mid_level, phase_3)
        noise = np.random.default_rng(int(abs(o * 100))).uniform(-noise_amp, noise_amp, phase_3)
        path[idx:idx + phase_3] = np.clip(base + noise, l, h)
        idx += phase_3

        # Phase 5: close drive
        path[idx:idx + phase_5] = np.linspace(mid_level, c, phase_5)
    else:
        # Phase 1: open -> push toward high (morning strength)
        push_target = h - (h - o) * 0.3
        path[idx:idx + phase_1] = np.linspace(o, push_target, phase_1)
        idx += phase_1

        # Phase 2: roll over from high toward mid
        path[idx:idx + phase_2] = np.linspace(push_target, (h + l) / 2, phase_2)
        idx += phase_2

        # Phase 3: range near middle
        mid_level = (l + c) / 2
        noise_amp = (h - l) * 0.05
        base = np.linspace((h + l) / 2, mid_level, phase_3)
        noise = np.random.default_rng(int(abs(o * 100))).uniform(-noise_amp, noise_amp, phase_3)
        path[idx:idx + phase_3] = np.clip(base + noise, l, h)
        idx += phase_3

        # Phase 5: close drive to daily close
        path[idx:idx + phase_5] = np.linspace(mid_level, c, phase_5)

    path = np.clip(path, l, h)
    path[0] = o
    path[-1] = c
    return path


def _generate_volume_profile(total_vol: float, n_bars: int = BARS_PER_DAY) -> np.ndarray:
    """U-shaped volume distribution (heavy at open and close)."""
    x = np.linspace(0, 1, n_bars)
    # U-shape: high at edges, low in middle
    profile = 2.0 * (x - 0.5) ** 2 + 0.3
    profile = profile / profile.sum() * total_vol
    return np.maximum(profile, 1).astype(int)


def synthesize_day(date_str: str, o: float, h: float, l: float, c: float,
                   volume: float, n_bars: int = BARS_PER_DAY) -> pd.DataFrame:
    """Synthesize 5-min bars for a single trading day."""
    closes = _generate_price_path(o, h, l, c, n_bars)
    volumes = _generate_volume_profile(volume, n_bars)

    opens = np.empty(n_bars)
    highs = np.empty(n_bars)
    lows = np.empty(n_bars)

    opens[0] = o
    for i in range(1, n_bars):
        opens[i] = closes[i - 1]

    bar_range = (h - l) / n_bars * 1.5
    for i in range(n_bars):
        bar_mid = (opens[i] + closes[i]) / 2
        highs[i] = min(h, max(opens[i], closes[i]) + bar_range * np.random.uniform(0, 0.5))
        lows[i] = max(l, min(opens[i], closes[i]) - bar_range * np.random.uniform(0, 0.5))

    # Generate timestamps: 9:30 to 16:00 ET, 5-min intervals
    base_time = pd.Timestamp(date_str) + pd.Timedelta(hours=9, minutes=30)
    times = [base_time + pd.Timedelta(minutes=5 * i) for i in range(n_bars)]

    return pd.DataFrame({
        "time_key": [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
        "open": np.round(opens, 4),
        "high": np.round(highs, 4),
        "low": np.round(lows, 4),
        "close": np.round(closes, 4),
        "volume": volumes,
    })


def synthesize_intraday(df_daily: pd.DataFrame,
                        n_bars: int = BARS_PER_DAY) -> pd.DataFrame:
    """Convert a daily OHLCV DataFrame to synthetic 5-min bars.

    Args:
        df_daily: DataFrame with columns [time_key, open, high, low, close, volume]
        n_bars: bars per day (default 78 for 5-min in US market)

    Returns:
        DataFrame with same columns, but n_bars rows per trading day.
    """
    days = []
    for _, row in df_daily.iterrows():
        date_str = str(row["time_key"])[:10]
        try:
            day_df = synthesize_day(
                date_str,
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row["volume"]), n_bars,
            )
            days.append(day_df)
        except Exception:
            continue

    if not days:
        return pd.DataFrame()

    result = pd.concat(days, ignore_index=True)
    return result


def load_or_synthesize_5min(symbol: str,
                            min_days: int = 200) -> Optional[pd.DataFrame]:
    """Smart loader: use real 5min data if sufficient, otherwise synthesize from daily.

    Args:
        symbol: 'TQQQ' or 'US.TQQQ'
        min_days: minimum trading days required

    Returns:
        DataFrame with 5min bars, or None
    """
    from data.downloader import load_5min, load_daily

    ticker = symbol.split(".")[-1] if "." in symbol else symbol

    real = load_5min(ticker)
    if real is not None:
        n_days = len(set(pd.to_datetime(real["time_key"]).dt.date))
        if n_days >= min_days:
            return real

    daily = load_daily(ticker)
    if daily is not None and len(daily) >= min_days:
        print(f"  Synthesizing {len(daily)} days of 5min data for {ticker}...")
        synth = synthesize_intraday(daily)
        print(f"  -> {len(synth)} synthetic 5min bars ({len(daily)} days)")
        return synth

    return real
