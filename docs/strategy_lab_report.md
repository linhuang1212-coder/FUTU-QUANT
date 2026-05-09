# $10K Strategy Lab Report (BIAS-CORRECTED v2)

Generated: 2026-05-08 14:56
Capital: $10,000 | Backtest: 10 years (2016-2026)
Strategies tested: 17

### Corrections Applied
- **Log returns** (vs simple returns) for all metrics
- **Transaction costs**: $1.00/trade + 5bps slippage + 3bps spread
- **Fat-tail diagnostics**: skewness, excess kurtosis, VaR, CVaR
- **Crisis stress tests**: COVID crash, L-shape recovery, rate hike, tariffs

## Overall Ranking

| Rank | Strategy | CAGR | Sharpe | Sortino | MaxDD | Vol | Skew | Kurt | VaR95 | Final |
|------|----------|------|--------|---------|-------|-----|------|------|-------|-------|
| 1 | Ensemble_SharpeWeight | 22.1% | 2.53 | 3.65 | -6.4% | 7.9% | -0.35 | 1.3 | -0.75% | $60,492 |
| 2 | Ensemble_InvVolWeight | 18.3% | 2.28 | 3.42 | -6.4% | 7.4% | -0.25 | 1.1 | -0.71% | $45,381 |
| 3 | Ensemble_EqualWeight | 17.8% | 2.22 | 3.30 | -6.7% | 7.4% | -0.26 | 1.2 | -0.74% | $43,929 |
| 4 | Trend_Vol_Target | 29.7% | 2.19 | 2.87 | -10.6% | 11.9% | -0.60 | 2.7 | -1.18% | $110,187 |
| 5 | Equal_Weight | 12.5% | 0.84 | 1.05 | -18.4% | 14.0% | -0.54 | 5.4 | -1.39% | $28,825 |
| 6 | Adaptive_AA | 11.3% | 0.82 | 1.01 | -18.5% | 13.0% | -0.69 | 4.5 | -1.29% | $27,678 |
| 7 | Factor_ETF_Rotation | 33.0% | 0.82 | 1.90 | -34.2% | 34.8% | 22.39 | 680.6 | -1.56% | $159,768 |
| 8 | InvVol_RiskParity | 10.2% | 0.81 | 1.01 | -15.9% | 12.0% | -0.49 | 4.6 | -1.25% | $24,074 |
| 9 | MinVariance | 7.0% | 0.70 | 0.86 | -18.5% | 9.6% | -0.35 | 6.0 | -0.99% | $18,394 |
| 10 | ETF_Momentum_Rotation | 17.8% | 0.70 | 0.81 | -31.6% | 23.5% | -1.43 | 17.7 | -2.11% | $35,084 |
| 11 | HRP_RiskParity | 6.9% | 0.67 | 0.83 | -18.3% | 10.0% | -0.47 | 4.9 | -1.00% | $18,260 |
| 12 | ETF_Momentum_Rotation | 12.7% | 0.63 | 0.76 | -27.1% | 18.9% | -1.09 | 10.3 | -1.79% | $23,551 |
| 13 | Dual_Momentum_GEM | 9.1% | 0.51 | 0.58 | -33.6% | 17.0% | -0.81 | 19.7 | -1.56% | $21,797 |
| 14 | XGBoost_ML | 8.0% | 0.37 | 0.45 | -40.5% | 20.8% | -0.61 | 15.6 | -1.91% | $18,580 |
| 15 | Mean_Reversion_ZScore | 3.4% | 0.36 | 0.25 | -22.5% | 9.3% | -1.87 | 79.7 | -0.59% | $13,580 |
| 16 | Mean_Reversion_ZScore | 1.2% | 0.18 | 0.09 | -19.5% | 7.0% | -2.78 | 79.4 | -0.35% | $11,198 |
| 17 | Pairs_Trading | -1.9% | -0.23 | -0.26 | -24.4% | 8.3% | -0.15 | 8.5 | -0.87% | $8,309 |

## Monthly Performance Distribution

| Strategy | Positive Months | Negative Months | Win Rate |
|----------|----------------|-----------------|----------|
| Ensemble_SharpeWeight | 85 | 23 | 79% |
| Ensemble_InvVolWeight | 84 | 24 | 78% |
| Ensemble_EqualWeight | 84 | 24 | 78% |
| Trend_Vol_Target | 88 | 22 | 80% |
| Equal_Weight | 76 | 32 | 70% |
| Adaptive_AA | 76 | 38 | 67% |
| Factor_ETF_Rotation | 82 | 34 | 71% |
| InvVol_RiskParity | 78 | 30 | 72% |
| MinVariance | 73 | 35 | 68% |
| ETF_Momentum_Rotation | 71 | 37 | 66% |

## Turnover & Transaction Cost Impact

| Strategy | Total TC Cost | Avg Daily Turnover | Rebalances |
|----------|-------------|-------------------|------------|
| Trend_Vol_Target | $9,506 | 0.1053 | 811 |

## Annual Return Breakdown

| Strategy | Y1 | Y2 | Y3 | Y4 | Y5 | Y6 | Y7 | Y8 | Y9 | Y10 |
|----------|------|------|------|------|------|------|------|------|------|------|
| Ensemble_SharpeWeight | +18.7% | +10.7% | +24.0% | +33.0% | +22.5% | +11.0% | +16.0% | +28.2% | +36.5% | +0.0% |
| Ensemble_InvVolWeight | +16.1% | +7.3% | +17.6% | +29.9% | +20.5% | +6.2% | +12.6% | +24.0% | +31.9% | +0.0% |
| Ensemble_EqualWeight | +15.7% | +6.8% | +16.9% | +30.0% | +19.9% | +5.8% | +12.3% | +23.3% | +31.7% | +0.0% |
| Trend_Vol_Target | +24.0% | +17.8% | +37.6% | +37.9% | +27.1% | +20.6% | +22.5% | +37.2% | +45.0% | +0.0% |
| Equal_Weight | +10.4% | -2.0% | +7.5% | +38.9% | +5.0% | +0.8% | +11.0% | +10.4% | +35.3% | +0.0% |
| Adaptive_AA | +12.3% | +3.9% | +6.5% | +12.4% | +26.9% | -5.5% | +2.8% | +21.7% | +14.5% | +0.0% |
| Factor_ETF_Rotation | +28.9% | +17.4% | +98.7% | +89.7% | +43.9% | -12.5% | +11.1% | +17.5% | +14.4% | +0.0% |
| InvVol_RiskParity | +10.1% | -0.5% | +9.9% | +27.7% | +1.4% | +0.1% | +8.7% | +10.8% | +26.6% | +0.0% |
| MinVariance | +4.8% | +0.4% | +13.2% | +15.8% | -4.7% | +6.4% | +5.5% | +12.6% | +10.3% | +0.0% |
| ETF_Momentum_Rotation | +14.8% | -2.6% | -5.8% | +44.2% | +25.3% | -5.4% | +11.3% | +12.0% | +85.2% | +0.0% |

## Validation Results (Top Strategies)

### K-Fold Time-Series Cross Validation (6 folds)

| Strategy | K-Fold Pass | Fold Sharpes | Robust? |
|----------|------------|-------------|---------|
| Ensemble_SharpeWeight | 6/6 | 2.14, 2.69, 3.05, 1.36, 2.31, 3.79 | YES |
| Ensemble_InvVolWeight | 6/6 | 2.13, 1.94, 3.10, 0.87, 2.26, 3.59 | YES |
| Ensemble_EqualWeight | 6/6 | 2.09, 1.79, 3.09, 0.80, 2.26, 3.51 | YES |
| Trend_Vol_Target | 6/6 | 1.69, 2.85, 2.23, 1.65, 2.02, 2.77 | YES |
| Equal_Weight | 6/6 | 0.64, 0.21, 1.87, 0.04, 1.36, 1.46 | YES |
| Adaptive_AA | 5/6 | 1.26, 0.26, 1.44, -0.16, 1.16, 1.15 | YES |
| Factor_ETF_Rotation | 5/6 | 2.08, 0.80, 1.02, -0.42, 1.31, 1.78 | YES |

### Monte Carlo + Confidence Intervals

| Strategy | Sharpe | MC Mean | PBO | 90% CI | Recovery |
|----------|--------|---------|-----|--------|----------|
| Ensemble_SharpeWeight | 2.53 | 2.53 | 100.0% | [1.89, 3.02] | 62d |
| Ensemble_InvVolWeight | 2.28 | 2.28 | 100.0% | [1.70, 2.91] | 141d |
| Ensemble_EqualWeight | 2.22 | 2.22 | 100.0% | [1.64, 2.70] | 143d |
| Trend_Vol_Target | 2.19 | 2.19 | 100.0% | [1.65, 2.73] | 139d |
| Equal_Weight | 0.84 | 0.84 | 97.4% | [0.39, 1.35] | 75d |
| Adaptive_AA | 0.82 | 0.82 | 100.0% | [0.30, 1.37] | 410d |
| Factor_ETF_Rotation | 0.82 | 0.82 | 81.0% | [0.53, 1.17] | 7d |

### Fat-Tail & Risk Diagnostics

| Strategy | Skewness | ExKurtosis | VaR95 | CVaR95 | VaR99 | CVaR99 | Normal? | Fat Tail? |
|----------|----------|-----------|-------|--------|-------|--------|---------|-----------|
| Ensemble_SharpeWeight | -0.35 | 1.3 | -0.75% | -1.08% | -1.35% | -1.65% | NO | OK |
| Ensemble_InvVolWeight | -0.25 | 1.1 | -0.71% | -0.98% | -1.15% | -1.45% | NO | OK |
| Ensemble_EqualWeight | -0.26 | 1.2 | -0.74% | -0.99% | -1.17% | -1.47% | NO | OK |
| Trend_Vol_Target | -0.60 | 2.7 | -1.18% | -1.73% | -2.01% | -2.75% | NO | OK |
| Equal_Weight | -0.54 | 5.4 | -1.39% | -2.19% | -2.57% | -3.45% | NO | WARNING |
| Adaptive_AA | -0.69 | 4.5 | -1.29% | -2.07% | -2.74% | -3.46% | NO | WARNING |
| Factor_ETF_Rotation | 22.39 | 680.6 | -1.56% | -2.66% | -2.94% | -4.87% | NO | WARNING |

### Stress Test Details

| Strategy | COVID Crash | COVID 6mo | L-Shape Yr | Rate Hike | Q4 2018 | Tariff 2025 |
|----------|-----------|----------|-----------|----------|---------|-------------|
| Ensemble_SharpeWeight | +2.5% | +14.3% | +22.5% | +8.3% | +3.7% | +0.0% |
| Ensemble_InvVolWeight | +3.4% | +14.6% | +20.5% | +8.5% | +2.0% | +0.0% |
| Ensemble_EqualWeight | +3.4% | +14.2% | +19.9% | +8.4% | +1.7% | +0.0% |
| Trend_Vol_Target | +0.8% | +14.4% | +27.1% | +8.0% | +7.4% | +0.0% |
| Equal_Weight | +2.9% | +4.7% | +5.0% | +4.7% | -4.6% | +0.0% |
| Adaptive_AA | +6.3% | +23.6% | +26.9% | +11.6% | +1.7% | +18.1% |
| Factor_ETF_Rotation | +1.0% | +24.2% | +43.9% | +1.2% | +0.1% | +10.3% |

### Crisis Behavior (Worst 10% of Days)

| Strategy | Mean Ret (worst 10%) | Vol (worst 10%) |
|----------|---------------------|-----------------|
| Ensemble_SharpeWeight | -0.855% | 0.332% |
| Ensemble_InvVolWeight | -0.798% | 0.274% |
| Ensemble_EqualWeight | -0.807% | 0.281% |
| Trend_Vol_Target | -1.347% | 0.595% |
| Equal_Weight | -1.665% | 0.795% |
| Adaptive_AA | -1.567% | 0.769% |
| Factor_ETF_Rotation | -1.978% | 1.295% |

## Recommendations

### Top 3 Strategies for $10K Portfolio (Post-Correction)

1. **Ensemble_SharpeWeight** — Sharpe 2.53, CAGR 22.1%, MaxDD -6.4%
2. **Ensemble_InvVolWeight** — Sharpe 2.28, CAGR 18.3%, MaxDD -6.4%
3. **Ensemble_EqualWeight** — Sharpe 2.22, CAGR 17.8%, MaxDD -6.7%

### Deployment Plan

1. Deploy Top 3 strategies to paper trading (Futu SIMULATE)
2. Run for 1-2 weeks, compare live NAV vs backtest expectations
3. If consistent, migrate to real account with scaled position sizes

### Risk Warnings

- Strategies with **excess kurtosis > 3** have fat tails — real losses may exceed VaR
- **Negative skewness** implies asymmetric downside risk
- **PBO > 5%** suggests possible overfitting
- Sharpe ratios above 2.0 after cost correction deserve extra scrutiny

---
*Generated by backtest/strategy_lab.py (bias-corrected v2)*