# 当前上下文

## 当前焦点
用 10 年合成日内数据完成了所有日内策略的重新验证，只保留经过 2511 天验证的策略。

## 最近完成
- **数据基础设施**：
  - data/downloader.py：Yahoo Finance 日线（10年）+ Alpha Vantage 5min（2年）+ Yahoo 5min（60天）
  - data/synthesizer.py：日线 → 5min 合成器（用 OHLC 约束生成 78 bar/天的价格路径）
  - run_intraday_scan.py / run_param_scan.py：优先读本地数据，不够时自动合成
- **10 年日内策略验证**（2016-04-22 ~ 2026-04-17，2511 天 x 2 标的）：
  - **ORB**: TQQQ Sharpe -1.54, SOXL -0.13 → **彻底失败**
  - **First_Pullback**: 没进 Top 20 → **失败**（之前 59 天假 Sharpe 8.4）
  - **Gap_Reversion**: TQQQ Sharpe -3.99 → **失败**
  - **Rebalance_2pm@TQQQ**: Sharpe **2.47**, Val Sharpe **2.73**, 胜率 75%, PF 3.1 → **PASS**
  - **VWAP_Trend@TQQQ**: Sharpe 0.61, Val Sharpe 1.04 → **弱 PASS**
- 实盘 INTRADAY_CONFIGS 已更新为仅保留验证通过的 2 个策略

## 实盘配置摘要
- 资本：$3,000，动态仓位
- **市场过滤**：VIX>28 禁入 / VIX>35 强平 / ADX>25 趋势确认 / EWMA vol target 18%
- 标的：US.TQQQ / US.SOXL
- **Swing 策略 (8个)**：
  - TQQQ: momentum, breakout, mean_reversion, multi_factor
  - SOXL: breakout, mean_reversion, rsi_reversal, multi_factor
- **日内策略 (2个, TQQQ only)**：
  - Rebalance_2pm (move>1.5%, 1:30pm入场, 1%止损) — 10年验证 Sharpe 2.47
  - VWAP_Trend (3bar确认, RSI>50) — 10年验证 Sharpe 0.61
- 运行：`python run_live.py` 或 `python run_live.py --dry-run`

## 本地数据
- `data_store/market_data/TQQQ_daily.csv`: 2,511 bars (2016-2026)
- `data_store/market_data/SOXL_daily.csv`: 2,511 bars (2016-2026)
- `data_store/market_data/TQQQ_5min.csv`: 4,559 bars (59 days real)
- `data_store/market_data/SOXL_5min.csv`: 4,559 bars (59 days real)
- 下载更多：`python -m data.downloader --av-key KEY --5min-only --months 24`

## 关键教训
- 59 天数据上 Sharpe 8+ 的策略在 10 年数据上完全失效（ORB, First_Pullback）
- Rebalance_2pm 反而是最稳定的：59 天 Sharpe 1.6 → 10年 Sharpe 2.47
- 杠杆 ETF 再平衡效应是有物理基础的（sponsor 必须在收盘前再平衡）

## 下一步
- 注册 Alpha Vantage API key，下载 2 年真实 5min 数据做交叉验证
- 配置 Telegram Bot 接收交易通知
- 观察实盘效果
