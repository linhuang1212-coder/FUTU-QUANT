# Factor Analysis Report

Generated: 2026-05-04 19:36

## Data
- Symbols: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, SPY, QQQ, IWM
- Period: 2021-05-05 to 2026-05-01
- Trading days: 1254
- Forward return: 5 days

## IC Analysis Summary

| Factor | IC Mean | IC Std | IC_IR | t-stat | IC>0% |
|--------|---------|--------|-------|--------|-------|
| VOL_60D | 0.0518 | 0.4046 | 0.1279 | 4.41 | 55.8% |
| VOL_20D | 0.0454 | 0.3901 | 0.1164 | 4.08 | 54.6% |
| MOM_12M | 0.0155 | 0.4067 | 0.0381 | 1.20 | 52.7% |
| MOM_6M | 0.0144 | 0.4224 | 0.0342 | 1.15 | 50.1% |
| IVR | 0.0036 | 0.3575 | 0.0101 | 0.32 | 47.9% |
| REVERSAL | 0.0036 | 0.3821 | 0.0094 | 0.33 | 50.7% |
| MOM_3M | -0.0040 | 0.4055 | -0.0098 | -0.34 | 50.3% |
| MOM_1M | -0.0096 | 0.3959 | -0.0243 | -0.85 | 48.0% |
| TURNOVER | -0.0323 | 0.3572 | -0.0903 | -3.06 | 45.3% |
| HV_RATIO | -0.0347 | 0.3477 | -0.0998 | -3.44 | 47.9% |

## Key Findings

**Top factors by IC_IR:** VOL_60D, VOL_20D, MOM_12M, MOM_6M

### Interpretation
- IC_IR > 0.5: Strong, stable predictive power
- IC_IR 0.1-0.5: Moderate, usable in composite
- IC_IR < 0.1: Weak, limited standalone value
- |t-stat| > 2: Statistically significant

## Recommendations for FUTU-QUANT

Factors with IC_IR > 0.1: **VOL_60D, VOL_20D**
These are candidates for Credit Spread symbol ranking and momentum rotation optimization.
