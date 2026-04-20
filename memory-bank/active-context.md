# 当前上下文

## 当前焦点
完成了8年真实5min数据下载和全面验证，确认所有日内策略在真实数据上失败。转向宏观趋势过滤+风险管理方向。

## 最近完成

### 8年真实5min数据下载 (2018-05 ~ 2026-04)
- TQQQ: 155,508 bars, 2,002 trading days
- SOXL: 155,508 bars, 2,002 trading days
- 使用 Futu API 按月分批下载，脚本: `_download_5min_futu.py`

### 8年真实5min验证结果（全部FAIL）
| 策略 | TQQQ Sharpe | SOXL Sharpe | 结论 |
|------|-----------|-----------|------|
| Rebalance_2pm | -0.119 | -0.233 | FAIL |
| VWAP_Trend | -1.195 | -0.996 | FAIL |
| ORB | -0.781 | -0.287 | FAIL |
| Momentum_Close | -1.638 | -0.607 | FAIL |
| Vol_Squeeze | 之前合成数据Sharpe=9.8 | 真实数据FAIL | 合成数据完全不可信 |

**核心教训：合成5min数据生成的策略100%不可信，只能用真实数据验证**

### 宏观趋势策略验证（10年真实日线）
| 策略 | 10yr Sharpe | MaxDD降低 | 定位 |
|------|-----------|---------|------|
| QQQ SMA200 趋势过滤 | 0.815 | 避开2022熊市(-80%降为0%) | 风险管理层 |
| 双动量轮换 | 0.771 | COVID期间Sharpe 1.745 | 标的选择层 |
| VIX自适应仓位 | 0.740 | MaxDD从60%降到38% | 仓位管理层 |

**结论：这些策略无法独立达到Sharpe>1.0，但作为风险管理叠加层非常有价值**

### 代码清理
- 删除3个失败策略文件: vwap_reversion.py, volatility_breakout.py, fast_ema_cross.py
- 删除3个过时脚本: run_new_strategy_scan.py, run_validation.py, _validate_real_5min.py
- run_live.py: 清空所有日内eval方法和INTRADAY_CONFIGS
- 新增: SMA200趋势过滤 + 双动量轮换 + VIX自适应仓位分级

## 实盘配置（最新）
- 资本：$3,000
- **全局风控层**：
  1. QQQ SMA200 过滤（QQQ < SMA200 → 全平仓不交易）
  2. VIX 连续分级仓位（<15: 95%, <20: 75%, <28: 50%, >=28: 0%）
  3. VIX > 35 强平
  4. ADX 趋势确认（仅趋势策略）
  5. EWMA 波动率目标 18%
  6. 回撤治理器（>20% 减半仓）
- **标的选择**：双动量轮换（每月排名 TQQQ vs SOXL）
- **Swing 策略**：
  - TQQQ: momentum, breakout, mean_reversion, multi_factor
  - SOXL: breakout, mean_reversion, rsi_reversal, multi_factor
- **日内策略**：全部禁用（8年真实数据全部 Sharpe 为负）
- 运行：`python run_live.py` 或 `python run_live.py --dry-run`

## 本地数据
- `QQQ_daily.csv`: 2,511 bars (2016-2026)
- `SPY_daily.csv`: 2,511 bars (2016-2026)
- `TQQQ_daily.csv`: 2,511 bars (2016-2026)
- `SOXL_daily.csv`: 2,511 bars (2016-2026)
- `QLD_daily.csv`: 2,511 bars
- `SPXL_daily.csv`: 2,511 bars
- `UPRO_daily.csv`: 2,511 bars
- `TNA_daily.csv`: 2,511 bars
- `TQQQ_5min.csv`: 155,508 bars (2018-05 ~ 2026-04, 8年真实)
- `SOXL_5min.csv`: 155,508 bars (2018-05 ~ 2026-04, 8年真实)

## GitHub + AutoDL
- GitHub: https://github.com/linhuang1212-coder/FUTU-QUANT (private)
- AutoDL: ssh -p 45630 root@connect.westd.seetacloud.com

## 关键教训
1. 合成5min数据完全不可信 — 策略在合成数据上Sharpe=9.8但真实数据为负
2. 3x杠杆ETF的日内alpha极难获取 — 交易成本+滑点吃掉微弱优势
3. SMA200趋势过滤是最有物理基础的风控（避开整个熊市）
4. VIX连续分级优于二元开关（降低回撤同时保留更多收益）
5. 双动量轮换在极端行情(COVID)表现优秀但中期不稳定
