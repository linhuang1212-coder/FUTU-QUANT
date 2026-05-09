# Turbo Validation 报告

生成时间: 2026-04-29 21:10:44
总耗时: 1531.6s
CPU 核心: 20 (使用 16 workers)

## 总览

| 策略 | 类型 | 总分 | 判定 | Sharpe | PBO | MC CI | 敏感性 | 压力 | DSR |
|------|------|------|------|--------|-----|-------|--------|------|-----|
| credit_spread | options | 63/100 | COND | 0.29 | 20% | [-0.80,3.41] | 0.51 | PASS | N/A |
| straddle_squeeze | options | 35/100 | FAIL | 0.30 | 33% | [-0.93,1.56] | 0.31 | PASS | N/A |
| earnings_spread | options | 73/100 | COND | 1.44 | 11% | [0.54,2.23] | 0.56 | PASS | N/A |
| wheel_csp | options | 53/100 | FAIL | 0.30 | 33% | [-0.75,1.92] | 0.51 | PASS | N/A |
| momentum | equity | 55/100 | FAIL | 0.74 | 4% | [-0.16,1.49] | N/A | PASS | N/A |
| mean_reversion | equity | 55/100 | FAIL | 0.50 | 13% | [-0.32,1.38] | N/A | PASS | N/A |
| breakout | equity | 15/100 | FAIL | 0.00 | 100% | N/A | N/A | PASS | N/A |
| rsi_reversal | equity | 43/100 | FAIL | 0.55 | 47% | [-0.31,1.33] | N/A | PASS | N/A |
| multi_factor | equity | 55/100 | FAIL | 0.83 | 4% | [-0.02,1.54] | N/A | PASS | N/A |
| momentum_rotation | etf | 25/100 | FAIL | 0.00 | 100% | N/A | N/A | PASS | N/A |

## 推荐部署

### PASS (推荐部署)
- (无)

### CONDITIONAL (需进一步验证)
- **credit_spread** (63/100) — CONDITIONAL — 需要 Paper Trading 进一步验证
- **earnings_spread** (73/100) — CONDITIONAL — 需要 Paper Trading 进一步验证

### FAIL (不建议部署)
- **straddle_squeeze** (35/100) — FAIL — 不建议部署
- **wheel_csp** (53/100) — FAIL — 不建议部署
- **momentum** (55/100) — FAIL — 不建议部署
- **mean_reversion** (55/100) — FAIL — 不建议部署
- **breakout** (15/100) — FAIL — 不建议部署
- **rsi_reversal** (43/100) — FAIL — 不建议部署
- **multi_factor** (55/100) — FAIL — 不建议部署
- **momentum_rotation** (25/100) — FAIL — 不建议部署

## 各策略详细结果

### credit_spread

**经济学假设**: Volatility Risk Premium: options are systematically overpriced relative to realized volatility. Selling OTM put spreads when IVR is elevated captures the IV-RV spread. Academic evidence: Coval & Shumway (2001), Bakshi & Kapadia (2003). Edge: systematic theta collection + mean reversion of IV.

**基准回测**: Sharpe = 0.29

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 10/10 | PASS | Volatility Risk Premium: options are systematically overpriced relative to reali... |
| 2. 含成本回测 Sharpe | 5/15 | FAIL | Sharpe = 0.29 |
| 3. CPCV PBO | 20/20 | PASS | PBO = 20.0% (45 paths) |
| 4. Monte Carlo CI | 5/15 | FAIL | Bootstrap CI [-0.80, 3.41] \| Noise median=2.05 |
| 5. 参数敏感性 | 8/15 | FAIL | Score = 0.51 \| Cliffs: target_delta, max_hold, tp_pct |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 63/100 — CONDITIONAL — 需要 Paper Trading 进一步验证**

#### CPCV 详情
- 路径数: 45
- PBO: 20.0%
- 平均 OOS Sharpe: 2.33
- 中位 OOS Sharpe: 2.48

#### Monte Carlo 详情
- Shuffle DD percentile: 36%
- Bootstrap Sharpe CI: [-0.80, 3.41]
- Noise Sharpe median: 2.05

#### 参数敏感性详情
| 参数 | Score | Plateau | Cliff |
|------|-------|---------|-------|
| spread_width | 0.20 | ✗ | — |
| target_delta | 0.33 | ✗ | ⚠ |
| max_hold | 0.75 | ✓ | ⚠ |
| tp_pct | 0.25 | ✗ | ⚠ |
| sl_pct | 1.00 | ✓ | — |

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 0 | $+0 | $+0 | — |
| COVID_Recovery | 2 | $+91 | $+0 | — |
| Rate_Hike_Bear | 8 | $+47 | $-2 | — |
| VIX_Spike_2024 | 0 | $+0 | $+0 | — |

---

### straddle_squeeze

**经济学假设**: Volatility mean reversion: periods of abnormally low realized volatility tend to be followed by volatility expansion. Buying straddles during BB squeeze captures the subsequent move. Academic evidence: volatility clustering (Mandelbrot 1963, Engle 1982 ARCH). Edge: buying vol cheaply when markets are complacent.

**基准回测**: Sharpe = 0.30

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 0/10 | FAIL | 未记录经济学假设 |
| 2. 含成本回测 Sharpe | 5/15 | FAIL | Sharpe = 0.30 |
| 3. CPCV PBO | 15/20 | PASS | PBO = 33.3% (45 paths) |
| 4. Monte Carlo CI | 0/15 | FAIL | Bootstrap CI [-0.93, 1.56] \| Noise median=-1.36 |
| 5. 参数敏感性 | 0/15 | FAIL | Score = 0.31 \| Cliffs: bb_percentile_threshold, max_holding_days, target_mult |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 35/100 — FAIL — 不建议部署**

#### CPCV 详情
- 路径数: 45
- PBO: 33.3%
- 平均 OOS Sharpe: 0.37
- 中位 OOS Sharpe: 0.15

#### Monte Carlo 详情
- Shuffle DD percentile: 84%
- Bootstrap Sharpe CI: [-0.93, 1.56]
- Noise Sharpe median: -1.36

#### 参数敏感性详情
| 参数 | Score | Plateau | Cliff |
|------|-------|---------|-------|
| bb_lookback | 0.33 | ✗ | — |
| bb_percentile_threshold | 0.25 | ✗ | ⚠ |
| max_holding_days | 0.33 | ✗ | ⚠ |
| target_mult | 0.33 | ✗ | ⚠ |

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 2 | $+310 | $+0 | — |
| COVID_Recovery | 1 | $+285 | $+0 | — |
| Rate_Hike_Bear | 2 | $+1,781 | $+0 | — |
| VIX_Spike_2024 | 2 | $+961 | $+0 | — |

---

### earnings_spread

**经济学假设**: Event-driven volatility: earnings announcements create information asymmetry and demand for options. IV expansion before events often exceeds realized moves, but buying early enough captures the IV run-up. Edge: systematic event premium harvesting with defined risk.

**基准回测**: Sharpe = 1.44

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 0/10 | FAIL | 未记录经济学假设 |
| 2. 含成本回测 Sharpe | 15/15 | PASS | Sharpe = 1.44 |
| 3. CPCV PBO | 20/20 | PASS | PBO = 11.1% (45 paths) |
| 4. Monte Carlo CI | 15/15 | PASS | Bootstrap CI [0.54, 2.23] \| Noise median=0.67 |
| 5. 参数敏感性 | 8/15 | FAIL | Score = 0.56 |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 73/100 — CONDITIONAL — 需要 Paper Trading 进一步验证**

#### CPCV 详情
- 路径数: 45
- PBO: 11.1%
- 平均 OOS Sharpe: 1.43
- 中位 OOS Sharpe: 1.48

#### Monte Carlo 详情
- Shuffle DD percentile: 32%
- Bootstrap Sharpe CI: [0.54, 2.23]
- Noise Sharpe median: 0.67

#### 参数敏感性详情
| 参数 | Score | Plateau | Cliff |
|------|-------|---------|-------|
| pre_event_days | 0.33 | ✗ | — |
| vol_threshold_mult | 0.33 | ✗ | — |
| max_cost_pct | 1.00 | ✓ | — |

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 0 | $+0 | $+0 | — |
| COVID_Recovery | 0 | $+0 | $+0 | — |
| Rate_Hike_Bear | 1 | $+633 | $+0 | — |
| VIX_Spike_2024 | 0 | $+0 | $+0 | — |

---

### wheel_csp

**经济学假设**: Cash-secured put selling: collect premium by selling puts on stocks you'd be willing to own. Combines volatility risk premium harvesting with value entry. Academic evidence: Coval & Shumway (2001). Edge: systematic income generation with stock ownership as downside.

**基准回测**: Sharpe = 0.30

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 0/10 | FAIL | 未记录经济学假设 |
| 2. 含成本回测 Sharpe | 10/15 | PASS | Sharpe = 0.30 |
| 3. CPCV PBO | 15/20 | PASS | PBO = 33.3% (45 paths) |
| 4. Monte Carlo CI | 5/15 | FAIL | Bootstrap CI [-0.75, 1.92] \| Noise median=1.61 |
| 5. 参数敏感性 | 8/15 | FAIL | Score = 0.51 \| Cliffs: target_delta, dte |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 53/100 — FAIL — 不建议部署**

#### CPCV 详情
- 路径数: 45
- PBO: 33.3%
- 平均 OOS Sharpe: 0.92
- 中位 OOS Sharpe: 0.29

#### Monte Carlo 详情
- Shuffle DD percentile: 96%
- Bootstrap Sharpe CI: [-0.75, 1.92]
- Noise Sharpe median: 1.61

#### 参数敏感性详情
| 参数 | Score | Plateau | Cliff |
|------|-------|---------|-------|
| target_delta | 0.20 | ✗ | ⚠ |
| dte | 0.50 | ✗ | ⚠ |
| tp_pct | 0.33 | ✗ | — |
| min_ivr | 1.00 | ✓ | — |

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 0 | $+0 | $+0 | — |
| COVID_Recovery | 1 | $+500 | $+0 | — |
| Rate_Hike_Bear | 6 | $+479 | $+0 | — |
| VIX_Spike_2024 | 0 | $+0 | $+0 | — |

---

### momentum

**经济学假设**: Time-series and cross-sectional momentum: winners continue winning. Jegadeesh & Titman (1993). Edge: systematic trend following with risk management.

**基准回测**: Sharpe = 0.74

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 0/10 | FAIL | 未记录经济学假设 |
| 2. 含成本回测 Sharpe | 15/15 | PASS | Sharpe = 0.74 |
| 3. CPCV PBO | 20/20 | PASS | PBO = 4.4% (45 paths) |
| 4. Monte Carlo CI | 5/15 | FAIL | Bootstrap CI [-0.16, 1.49] \| Noise median=0.45 |
| 5. 参数敏感性 | 0/15 | FAIL | 未执行 |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 55/100 — FAIL — 不建议部署**

#### CPCV 详情
- 路径数: 45
- PBO: 4.4%
- 平均 OOS Sharpe: 0.76
- 中位 OOS Sharpe: 0.76

#### Monte Carlo 详情
- Shuffle DD percentile: 16%
- Bootstrap Sharpe CI: [-0.16, 1.49]
- Noise Sharpe median: 0.45

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 5 | $+221 | $-110 | — |
| COVID_Recovery | 5 | $+221 | $-110 | — |
| Rate_Hike_Bear | 10 | $-17 | $-286 | — |
| VIX_Spike_2024 | 3 | $+349 | $+0 | — |

---

### mean_reversion

**经济学假设**: Short-term mean reversion: overreaction to news causes temporary price displacement. Poterba & Summers (1988). Edge: statistical tendency for prices to revert to moving averages.

**基准回测**: Sharpe = 0.50

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 0/10 | FAIL | 未记录经济学假设 |
| 2. 含成本回测 Sharpe | 15/15 | PASS | Sharpe = 0.50 |
| 3. CPCV PBO | 20/20 | PASS | PBO = 13.3% (45 paths) |
| 4. Monte Carlo CI | 5/15 | FAIL | Bootstrap CI [-0.32, 1.38] \| Noise median=0.80 |
| 5. 参数敏感性 | 0/15 | FAIL | 未执行 |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 55/100 — FAIL — 不建议部署**

#### CPCV 详情
- 路径数: 45
- PBO: 13.3%
- 平均 OOS Sharpe: 0.77
- 中位 OOS Sharpe: 0.60

#### Monte Carlo 详情
- Shuffle DD percentile: 6%
- Bootstrap Sharpe CI: [-0.32, 1.38]
- Noise Sharpe median: 0.80

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 7 | $-588 | $-548 | — |
| COVID_Recovery | 8 | $-272 | $-548 | — |
| Rate_Hike_Bear | 11 | $+474 | $-184 | — |
| VIX_Spike_2024 | 6 | $+258 | $-100 | — |

---

### breakout

**经济学假设**: Trend continuation after range breakout: consolidation patterns resolve with momentum. Edge: capturing the transition from low to high volatility regimes.

**基准回测**: Sharpe = 0.00

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 0/10 | FAIL | 未记录经济学假设 |
| 2. 含成本回测 Sharpe | 0/15 | FAIL | Sharpe = 0.00 |
| 3. CPCV PBO | 0/20 | FAIL | PBO = 100.0% (45 paths) |
| 4. Monte Carlo CI | 0/15 | FAIL | 未执行 |
| 5. 参数敏感性 | 0/15 | FAIL | 未执行 |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 15/100 — FAIL — 不建议部署**

#### CPCV 详情
- 路径数: 45
- PBO: 100.0%
- 平均 OOS Sharpe: 0.00
- 中位 OOS Sharpe: 0.00

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 0 | $+0 | $+0 | — |
| COVID_Recovery | 0 | $+0 | $+0 | — |
| Rate_Hike_Bear | 0 | $+0 | $+0 | — |
| VIX_Spike_2024 | 0 | $+0 | $+0 | — |

---

### rsi_reversal

**经济学假设**: RSI oversold/overbought reversal: extreme RSI readings signal exhaustion. Wilder (1978). Edge: contrarian entry at statistical extremes.

**基准回测**: Sharpe = 0.55

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 0/10 | FAIL | 未记录经济学假设 |
| 2. 含成本回测 Sharpe | 15/15 | PASS | Sharpe = 0.55 |
| 3. CPCV PBO | 8/20 | FAIL | PBO = 46.7% (45 paths) |
| 4. Monte Carlo CI | 5/15 | FAIL | Bootstrap CI [-0.31, 1.33] \| Noise median=0.50 |
| 5. 参数敏感性 | 0/15 | FAIL | 未执行 |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 43/100 — FAIL — 不建议部署**

#### CPCV 详情
- 路径数: 45
- PBO: 46.7%
- 平均 OOS Sharpe: 0.02
- 中位 OOS Sharpe: 0.05

#### Monte Carlo 详情
- Shuffle DD percentile: 52%
- Bootstrap Sharpe CI: [-0.31, 1.33]
- Noise Sharpe median: 0.50

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 4 | $-231 | $-166 | — |
| COVID_Recovery | 4 | $-113 | $-83 | — |
| Rate_Hike_Bear | 12 | $-317 | $-499 | — |
| VIX_Spike_2024 | 4 | $+133 | $-101 | — |

---

### multi_factor

**经济学假设**: Factor combination: multiple weak signals aggregated produce robust composite signal. Edge: diversification across alpha sources reduces single-factor risk.

**基准回测**: Sharpe = 0.83

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 0/10 | FAIL | 未记录经济学假设 |
| 2. 含成本回测 Sharpe | 15/15 | PASS | Sharpe = 0.83 |
| 3. CPCV PBO | 20/20 | PASS | PBO = 4.4% (45 paths) |
| 4. Monte Carlo CI | 5/15 | FAIL | Bootstrap CI [-0.02, 1.54] \| Noise median=0.86 |
| 5. 参数敏感性 | 0/15 | FAIL | 未执行 |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 55/100 — FAIL — 不建议部署**

#### CPCV 详情
- 路径数: 45
- PBO: 4.4%
- 平均 OOS Sharpe: 0.89
- 中位 OOS Sharpe: 0.89

#### Monte Carlo 详情
- Shuffle DD percentile: 9%
- Bootstrap Sharpe CI: [-0.02, 1.54]
- Noise Sharpe median: 0.86

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 5 | $+105 | $-101 | — |
| COVID_Recovery | 4 | $+112 | $-112 | — |
| Rate_Hike_Bear | 10 | $-90 | $-423 | — |
| VIX_Spike_2024 | 4 | $+431 | $-88 | — |

---

### momentum_rotation

**经济学假设**: Cross-sectional momentum: assets with strong recent returns continue to outperform. 12M-1M momentum avoids short-term reversal noise. SMA200 trend filter reduces drawdowns during bear markets. Academic evidence: Jegadeesh & Titman (1993), Moskowitz et al. (2012). Edge: systematic risk premia harvesting across asset classes.

**基准回测**: Sharpe = 0.00

| Gate | 分数 | 状态 | 详情 |
|------|------|------|------|
| 1. 经济学假设 | 10/10 | PASS | Cross-sectional momentum: assets with strong recent returns continue to outperfo... |
| 2. 含成本回测 Sharpe | 0/15 | FAIL | Sharpe = 0.00 |
| 3. CPCV PBO | 0/20 | FAIL | PBO = 100.0% (45 paths) |
| 4. Monte Carlo CI | 0/15 | FAIL | 未执行 |
| 5. 参数敏感性 | 0/15 | FAIL | 未执行 |
| 6. 压力测试 | 15/15 | PASS | 所有压力期通过 |
| 7. DSR 显著性 | 0/10 | FAIL | 未执行 |

**总分: 25/100 — FAIL — 不建议部署**

#### CPCV 详情
- 路径数: 45
- PBO: 100.0%
- 平均 OOS Sharpe: 0.00
- 中位 OOS Sharpe: 0.00

#### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| Rate_Hike_Bear | 0 | $+0 | $+0 | — |
| VIX_Spike_2024 | 0 | $+0 | $+0 | — |

---

## 组合协同分析

组合 Sharpe: 1.45
组合最大回撤: $-1,378
参与策略数: 2

策略相关性矩阵:

| | credit_spread | earnings_spread |
|---|---|---|
| credit_spread | 1.00 | 0.02 |
| earnings_spread | 0.02 | 1.00 |

## 结论与建议


共 2 个策略为 CONDITIONAL，建议先进行 Paper Trading:
- **credit_spread** (得分 63/100)
- **earnings_spread** (得分 73/100)

共 8 个策略未通过验证，不建议部署:
- **straddle_squeeze** (得分 35/100)
- **wheel_csp** (得分 53/100)
- **momentum** (得分 55/100)
- **mean_reversion** (得分 55/100)
- **breakout** (得分 15/100)
- **rsi_reversal** (得分 43/100)
- **multi_factor** (得分 55/100)
- **momentum_rotation** (得分 25/100)

组合运行时预期 Sharpe 为 1.45，最大回撤 $-1,378。
策略间平均相关性: 0.02 (低相关，组合效果好)。
