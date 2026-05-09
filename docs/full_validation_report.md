# 完整量化验证管线报告

生成时间: 2026-04-29 02:10
资本: $3,000

## 总览

| 策略 | 总分 | 判定 |
|------|------|------|
| credit_spread | 93/100 | PASS — 推荐小仓位部署 |
| wheel_csp | 55/100 | FAIL — 不建议部署 |

## credit_spread

| Gate | 检验项目 | 得分 | 通过 | 详情 |
|------|---------|------|------|------|
| 1 | 经济学假设 | 10/10 | ✓ | Volatility Risk Premium: options are systematically overpriced relative to reali... |
| 2 | 含成本回测 Sharpe | 15/15 | ✓ | Sharpe = 2.77 |
| 3 | CPCV PBO | 20/20 | ✓ | PBO = 24.4% (45 paths) |
| 4 | Monte Carlo CI | 15/15 | ✓ | Bootstrap CI [2.16, 6.16] | Noise median=2.04 |
| 5 | 参数敏感性 | 8/15 | ✗ | Score = 0.47 | Cliffs: target_delta |
| 6 | 压力测试 | 15/15 | ✓ | 所有压力期通过 |
| 7 | DSR 显著性 | 10/10 | ✓ | DSR = 1.687, p = 0.011 |

### CPCV 详情
- 路径数: 45
- PBO: 24.4%
- 平均 OOS Sharpe: 2.64
- 中位 OOS Sharpe: 2.77

### Monte Carlo 详情
- Shuffle DD percentile: 100%
- Bootstrap Sharpe CI: [2.16, 6.16]
- Noise Sharpe median: 2.04

### 参数敏感性详情
| 参数 | Score | Plateau | Cliff |
|------|-------|---------|-------|
| spread_width | 0.25 | ✗ | — |
| | values: [1.0, 2.5, 5.0, 7.5, 10.0] | | |
| | sharpes: ['-999.00', '-0.61', '2.77', '2.33', '2.33'] | | |
| target_delta | 0.20 | ✗ | ⚠ |
| | values: [0.15, 0.2, 0.25, 0.3, 0.35] | | |
| | sharpes: ['-0.41', '-0.37', '-0.60', '2.77', '1.05'] | | |
| tp_pct | 0.67 | ✓ | — |
| | values: [0.3, 0.5, 0.75] | | |
| | sharpes: ['2.21', '2.77', '2.40'] | | |
| sl_pct | 1.00 | ✓ | — |
| | values: [1.0, 1.5, 2.0, 3.0] | | |
| | sharpes: ['2.77', '2.77', '2.77', '2.77'] | | |
| min_ivr | 0.25 | ✗ | — |
| | values: [40, 50, 60, 70] | | |
| | sharpes: ['2.32', '1.61', '2.77', '1.85'] | | |

### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 0 | $+0 | $+0 | — |
| COVID_Recovery | 2 | $+95 | $+0 | — |
| Rate_Hike_Bear | 6 | $+23 | $-1 | — |
| VIX_Spike_2024 | 0 | $+0 | $+0 | — |

## wheel_csp

| Gate | 检验项目 | 得分 | 通过 | 详情 |
|------|---------|------|------|------|
| 1 | 经济学假设 | 10/10 | ✓ | Cash-Secured Put exploits the same VRP as credit spreads but accepts assignment ... |
| 2 | 含成本回测 Sharpe | 5/15 | ✗ | Sharpe = 0.01 |
| 3 | CPCV PBO | 20/20 | ✓ | PBO = 13.3% (45 paths) |
| 4 | Monte Carlo CI | 5/15 | ✗ | Bootstrap CI [-0.33, 0.98] | Noise median=0.29 |
| 5 | 参数敏感性 | 0/15 | ✗ | Score = 0.23 | Cliffs: target_delta, dte, tp_pct |
| 6 | 压力测试 | 15/15 | ✓ | 所有压力期通过 |
| 7 | DSR 显著性 | 0/10 | ✗ | DSR = -1.006, p = 1.000 |

### CPCV 详情
- 路径数: 45
- PBO: 13.3%
- 平均 OOS Sharpe: 1.44
- 中位 OOS Sharpe: 1.29

### Monte Carlo 详情
- Shuffle DD percentile: 40%
- Bootstrap Sharpe CI: [-0.33, 0.98]
- Noise Sharpe median: 0.29

### 参数敏感性详情
| 参数 | Score | Plateau | Cliff |
|------|-------|---------|-------|
| target_delta | 0.20 | ✗ | ⚠ |
| | values: [0.2, 0.25, 0.3, 0.35, 0.4] | | |
| | sharpes: ['1.06', '0.94', '-0.52', '1.53', '-0.52'] | | |
| dte | 0.25 | ✗ | ⚠ |
| | values: [14, 21, 30, 45] | | |
| | sharpes: ['3.17', '1.53', '-0.01', '0.21'] | | |
| tp_pct | 0.25 | ✗ | ⚠ |
| | values: [0.25, 0.5, 0.75, 1.0] | | |
| | sharpes: ['0.39', '1.53', '2.27', '0.49'] | | |

### 压力测试详情
| 事件 | 交易数 | PnL | MaxDD | Veto |
|------|--------|-----|-------|------|
| COVID_Crash | 0 | $+0 | $+0 | — |
| COVID_Recovery | 2 | $+12 | $+0 | — |
| Rate_Hike_Bear | 8 | $+35 | $+0 | — |
| VIX_Spike_2024 | 0 | $+0 | $+0 | — |