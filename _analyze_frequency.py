"""Analyze actual trading frequency and capital utilization."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data_store/market_data")
POOL = ["TQQQ", "SOXL", "UPRO", "TECL"]


def load(sym):
    return pd.read_csv(DATA_DIR / f"{sym}_daily.csv", parse_dates=["time_key"]).sort_values("time_key").reset_index(drop=True)


def main():
    all_raw = {s: load(s) for s in POOL}
    qqq = load("QQQ").set_index("time_key")

    ref = all_raw["TQQQ"]
    m = (ref["time_key"] >= "2023-04-01") & (ref["time_key"] <= "2026-04-17")
    dates = ref[m]["time_key"].tolist()

    print(f"{'=' * 70}")
    print(f"  Trading Frequency & Capital Utilization Analysis (3yr)")
    print(f"{'=' * 70}")

    # Track daily state
    in_pos_days = 0
    flat_days = 0
    trades = 0
    holding = False
    flat_streaks = []
    pos_streaks = []
    cur_flat = 0
    cur_pos = 0

    for di, day in enumerate(dates):
        if di < 63:
            flat_days += 1
            cur_flat += 1
            continue

        mom = {}
        for s in POOL:
            before = all_raw[s][all_raw[s]["time_key"] <= day]
            if len(before) < 63:
                continue
            c = before["close"].values
            mom[s] = 0.5 * (c[-1] / c[-21] - 1) + 0.5 * (c[-1] / c[-63] - 1)

        any_positive = any(v > 0 for v in mom.values())

        qb = qqq.loc[:day]
        sma200_ok = len(qb) >= 200 and qb["close"].iloc[-1] > qb["close"].iloc[-200:].mean()

        can_trade = any_positive and sma200_ok

        if can_trade:
            if not holding:
                holding = True
                trades += 1
                if cur_flat > 0:
                    flat_streaks.append(cur_flat)
                cur_flat = 0
            in_pos_days += 1
            cur_pos += 1
        else:
            if holding:
                holding = False
                trades += 1
                if cur_pos > 0:
                    pos_streaks.append(cur_pos)
                cur_pos = 0
            flat_days += 1
            cur_flat += 1

    if cur_flat > 0:
        flat_streaks.append(cur_flat)
    if cur_pos > 0:
        pos_streaks.append(cur_pos)

    total = in_pos_days + flat_days
    round_trips = trades // 2

    print(f"\n  Total trading days:     {total}")
    print(f"  Days IN position:       {in_pos_days} ({in_pos_days/total*100:.1f}%)")
    print(f"  Days FLAT (cash):       {flat_days} ({flat_days/total*100:.1f}%)")
    print(f"  Round-trip trades:      {round_trips}")
    print(f"  Avg holding period:     {np.mean(pos_streaks):.1f} days" if pos_streaks else "")
    print(f"  Median holding period:  {np.median(pos_streaks):.1f} days" if pos_streaks else "")

    print(f"\n  Flat streak analysis:")
    print(f"    Mean:    {np.mean(flat_streaks):.1f} days")
    print(f"    Median:  {np.median(flat_streaks):.1f} days")
    print(f"    Max:     {max(flat_streaks)} days")
    print(f"    >5 days: {sum(1 for x in flat_streaks if x > 5)} times")
    print(f"    >20 days:{sum(1 for x in flat_streaks if x > 20)} times")

    # Monthly breakdown
    print(f"\n{'─' * 70}")
    print(f"  Monthly Capital Utilization (% of days in position)")
    print(f"{'─' * 70}")

    month_data = {}
    for di, day in enumerate(dates):
        ym = day.strftime("%Y-%m")
        if ym not in month_data:
            month_data[ym] = {"days": 0, "in_pos": 0}
        month_data[ym]["days"] += 1

        if di < 63:
            continue

        mom = {}
        for s in POOL:
            before = all_raw[s][all_raw[s]["time_key"] <= day]
            if len(before) < 63:
                continue
            c = before["close"].values
            mom[s] = 0.5 * (c[-1] / c[-21] - 1) + 0.5 * (c[-1] / c[-63] - 1)

        qb = qqq.loc[:day]
        sma200_ok = len(qb) >= 200 and qb["close"].iloc[-1] > qb["close"].iloc[-200:].mean()

        if any(v > 0 for v in mom.values()) and sma200_ok:
            month_data[ym]["in_pos"] += 1

    for ym in sorted(month_data.keys()):
        d = month_data[ym]
        util = d["in_pos"] / d["days"] * 100 if d["days"] > 0 else 0
        bar = "#" * int(util / 3)
        print(f"  {ym}  {d['in_pos']:>3}/{d['days']:<3} days  {util:>5.0f}%  {bar}")

    # Capital efficiency
    print(f"\n{'─' * 70}")
    print(f"  Capital Efficiency Summary")
    print(f"{'─' * 70}")
    utilization = in_pos_days / total * 100
    print(f"  Overall utilization: {utilization:.1f}% (money working {in_pos_days} of {total} days)")
    if utilization < 50:
        idle_pct = 100 - utilization
        print(f"  >>> WARNING: Cash idle {idle_pct:.0f}% of the time!")
        print(f"  >>> ${3000 * idle_pct / 100:.0f} sitting idle on average")


if __name__ == "__main__":
    main()
