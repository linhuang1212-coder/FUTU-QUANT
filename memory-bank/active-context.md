# 当前上下文

## 当前焦点
完成了全面的新策略研究 + 分段验证（10yr/5yr/3yr + 压力测试 + Walk-Forward），在 AutoDL 服务器上运行。

## 最近完成

### 分段验证框架 (run_segmented_validation.py)
- 新建统一验证框架：10yr/5yr/3yr 时间分段 + COVID/加息/AI牛市压力测试 + 滚动 WF
- 严格通过标准：三段 Sharpe 全部 > 1.0 + 压力期 MaxDD < 40% + WF 一致性 >= 60%
- 支持 swing 和 intraday 两种模式
- 在 AutoDL 服务器 (GPU 云) 上远程执行

### 新数据下载
- 新增 4 个标的日线数据：QLD、SPXL、UPRO、TNA（各 2511 bars，2016-2026）
- 共 6 个标的参与验证

### Swing 策略研究结果（全部 FAIL）
| 策略 | 最佳 10yr Sharpe | 结论 |
|------|-----------------|------|
| S1: SMA200 趋势过滤 | 0.74 (TQQQ) | FAIL |
| S2: 双均线金叉 | 0.76 (TQQQ) | FAIL |
| S4: RSI(2) Connors | 0.83 (SOXL) | FAIL |
| S5: Turnaround Tuesday | 0.64 (SOXL) | FAIL |
| Momentum 系列 | 0.82 (TQQQ) | FAIL |
| Mean Reversion 系列 | 0.62 (SOXL) | FAIL |
| Breakout 系列 | 0.44 (SOXL) | FAIL |
| RSI Reversal | 0.68 (SOXL) | FAIL |

**结论：单一 swing 策略在杠杆 ETF 上难以达到 Sharpe > 1.0**

### 日内策略研究结果（14 个 PASS）
| 策略 | 标的 | 10yr Sharpe | 5yr | 3yr | WF |
|------|------|-----------|-----|-----|-----|
| Rebalance_2pm (1.5%/48bar) | TQQQ | 2.47 | 2.67 | 2.69 | 100% |
| Rebalance_2pm (1.0%/48bar) | TQQQ | 3.25 | 3.56 | 3.73 | 100% |
| Rebalance_2pm (1.0%/48bar) | QLD | 3.61 | 4.03 | 3.26 | 100% |
| Rebalance_2pm (1.0%/48bar) | SPXL | 3.01 | 3.70 | 3.60 | 100% |
| I2_Afternoon_Ext (1.5%/36bar) | TQQQ | 3.89 | 4.42 | 3.82 | 100% |
| I2_Afternoon_Ext (1.5%/36bar) | SOXL | 4.28 | 3.98 | 3.13 | 100% |
| I2_Afternoon_Ext (1.5%/36bar) | TNA | 3.77 | 4.23 | 3.93 | 100% |
| I3_Vol_Squeeze (0.5/24bar/1%) | TQQQ | 9.77 | 10.43 | 10.75 | 100% |
| I3_Vol_Squeeze (0.5/24bar/1%) | SOXL | 9.84 | 10.02 | 10.17 | 100% |
| I3_Vol_Squeeze (0.5/24bar/1%) | 全部6标的 | 8.5~9.8 | 全部 | 全部 | 100% |

**注意：I3_Vol_Squeeze Sharpe 异常高(9~10)，需在真实 5min 数据上验证（合成数据可能高估）**

## 实盘配置摘要（更新后）
- 资本：$3,000，动态仓位
- **市场过滤**：VIX>28 禁入 / VIX>35 强平 / ADX>25 趋势确认 / EWMA vol target 18%
- 标的：US.TQQQ / US.SOXL
- **Swing 策略 (8个)**：
  - TQQQ: momentum, breakout, mean_reversion, multi_factor
  - SOXL: breakout, mean_reversion, rsi_reversal, multi_factor
- **日内策略 (5个, TQQQ+SOXL)**：
  - TQQQ: Rebalance_2pm (1.5%/48bar), Afternoon_Ext (1.5%/36bar), Vol_Squeeze (0.3/24bar)
  - SOXL: Afternoon_Ext (1.5%/36bar), Vol_Squeeze (0.3/24bar)
- 运行：`python run_live.py` 或 `python run_live.py --dry-run`

## 本地数据
- `TQQQ_daily.csv`: 2,511 bars (2016-2026)
- `SOXL_daily.csv`: 2,511 bars (2016-2026)
- `QLD_daily.csv`: 2,511 bars (2016-2026)
- `SPXL_daily.csv`: 2,511 bars (2016-2026)
- `UPRO_daily.csv`: 2,511 bars (2016-2026)
- `TNA_daily.csv`: 2,511 bars (2016-2026)
- `TQQQ_5min.csv`: 4,559 bars (59 days real)
- `SOXL_5min.csv`: 4,559 bars (59 days real)

## GitHub + AutoDL 部署
- GitHub: https://github.com/linhuang1212-coder/FUTU-QUANT (private)
- AutoDL: ssh -p 45630 root@connect.westd.seetacloud.com
- 远程测试：python _remote_exec.py "command"
- 数据上传：python _upload_data.py

## 关键教训
- 所有 swing 策略在杠杆 ETF 上 Sharpe 均 < 1.0（最高 0.83）
- 日内再平衡效应策略是最稳健的（物理基础：sponsor 再平衡买盘）
- 合成 5min 数据上 I3_Vol_Squeeze Sharpe 异常高，需要真实数据验证
- 提前入场（1pm instead of 1:30pm）显著提升 Sharpe（3.89 vs 2.47）
- 6 个标的中 SOXL 表现最差（日内策略大量 FAIL）

## 下一步
- 注册 Alpha Vantage API key，下载 2 年真实 5min 数据验证 Vol_Squeeze
- 配置 Telegram Bot 接收交易通知
- 考虑期权策略：买入看涨替代持股 + Covered Call
