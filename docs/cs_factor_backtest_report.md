# Credit Spread Factor Ranking Backtest Report

Generated: 2026-05-05 21:23

## Configuration

- **Symbols**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, SPY, QQQ, IWM (10 total)
- **Hold Periods**: 21D, 45D
- **Rebalance**: Monthly
- **Selection**: Top-3 per month
- **Delta**: 0.3 (strike placed ~0.52 sigma below spot)
- **Spread Width**: $2.5
- **Credit**: $0.750 per share (30% of width, assumes IVR>60)
- **Profit Take**: 50% of credit received

## Factor Weights

- Momentum (3M): 40%
- Volatility (60D): 30%
- IVR Proxy: 30%

## Results (21-Day Hold)

| Group | Trades | Win Rate | Avg P&L | Total P&L | Sharpe |
|-------|-------:|--------:|--------:|----------:|-------:|
| Random (baseline) | 168 | 87.3% | $21.70 | $3646.27 | 1.65 |
| IVR-only | 168 | 87.5% | $22.27 | $3740.86 | 1.72 |
| **Factor Ranked** | 168 | 87.5% | $22.49 | $3778.36 | 1.73 |

## Results (45-Day Hold)

| Group | Trades | Win Rate | Avg P&L | Total P&L | Sharpe |
|-------|-------:|--------:|--------:|----------:|-------:|
| Random (baseline) | 163 | 92.6% | $27.67 | $4515.10 | 2.64 |
| IVR-only | 163 | 92.6% | $27.84 | $4537.50 | 2.66 |
| **Factor Ranked** | 163 | 92.0% | $26.99 | $4400.00 | 2.49 |

## Summary & Conclusion

**21D Hold**: Factor WR 87.5% (vs Random +0.2pp, vs IVR +0.0pp) | Sharpe 1.73

**45D Hold**: Factor WR 92.0% (vs Random -0.6pp, vs IVR -0.6pp) | Sharpe 2.49

### Verdict

Factor ranking shows mixed results: slight improvement on shorter hold periods but no significant edge on longer holds. The improvement is within noise for this sample size. **Marginal benefit at best.**

### Key Insights

1. All groups achieve high win rates (85-93%) due to the short-put structure at 0.30 delta
2. The difference between random and factor-ranked is <1pp, statistically insignificant
3. With 10 highly-correlated mega-cap symbols, stock selection adds minimal alpha
4. IVR-only selection performs nearly identically to the full multi-factor model
5. Strategy profitability depends more on credit collected vs tail risk than on symbol selection

## Methodology

1. Monthly rebalance: compute factors using trailing data only (no look-ahead bias)
2. Bull Put Spread simulation: sell put at ~0.52 sigma OTM (matching delta=0.30)
3. Strike uses realized 60D volatility to determine OTM distance
4. Win condition: price stays above short strike at expiry
5. Early exit: 50% profit take if price moves sufficiently above strike mid-period
6. Random baseline: averaged over 100 trials for stability
