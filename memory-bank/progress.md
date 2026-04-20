# 项目进度

## 已完成

### 第一阶段：核心框架
- [x] 项目脚手架（config, requirements, .gitignore, README）
- [x] EventBus 事件总线
- [x] Order/Signal/Position 数据模型
- [x] Logger 日志系统 + YAML 配置加载
- [x] TechnicalIndicators 技术指标模块
- [x] BaseStrategy 抽象基类 + 5 个内置策略
- [x] RiskManager 风控引擎 + PdtGuard PDT 规则守护
- [x] TelegramNotifier 通知模块
- [x] MarketData 行情接口 + HistoryManager 缓存
- [x] Backtester 回测引擎 + BacktestReport 报告
- [x] Memory Bank 上下文持久化

### 第二阶段：策略研究和回测增强
- [x] 回测报告增强：Sharpe/Sortino/Calmar/CAGR/月度收益/Buy&Hold 对比
- [x] 参数扫描引擎：网格搜索 + Walk-Forward 验证
- [x] 多时间框架支持：K线重采样 + MultiTimeframeStrategy
- [x] 新增 RsiReversalStrategy、MultiFactorStrategy
- [x] 模拟盘验证管线

### 第三阶段：多策略组合实盘系统
- [x] MultiStrategyTrader（多标的多策略 + Sharpe加权信号仲裁）
- [x] PDT 双层交易系统（swing + intraday）
- [x] live.yaml 配置驱动

### 第四阶段：市场状态过滤 + 动态仓位
- [x] VolatilityTargetManager + MarketRegime 数据类
- [x] VIX 过滤器 + ADX 趋势确认
- [x] EWMA 波动率目标 + Drawdown Governor

### 第五阶段：全面验证 + 数据基础设施
- [x] data/downloader.py：Yahoo Finance 日线 + 统一加载接口
- [x] data/synthesizer.py：日线→5min 合成器
- [x] 下载 8 标的 10 年日线数据（QQQ/SPY/TQQQ/SOXL/QLD/SPXL/UPRO/TNA）
- [x] 8 年真实 5min 数据下载（2018-05 ~ 2026-04，TQQQ/SOXL 各 155K bars）
- [x] run_segmented_validation.py：10yr/5yr/3yr 分段验证框架
- [x] GitHub 仓库 + AutoDL 远程测试环境
- [x] Swing 策略研究（S1~S5）：**全部 FAIL**（最高 Sharpe 0.83）
- [x] 日内策略研究（I1~I3 + 已有策略）：合成数据上 PASS（Sharpe 3~10）
- [x] **真实 8 年 5min 数据验证：所有日内策略全部 FAIL**（Sharpe 全为负）
- [x] 期权策略研究报告

### 第六阶段：宏观趋势过滤（当前）
- [x] QQQ SMA200 趋势过滤验证（10yr Sharpe 0.815，有效避开熊市）
- [x] 双动量月度轮换验证（10yr Sharpe 0.771，COVID Sharpe 1.745）
- [x] VIX 自适应连续仓位分级（MaxDD 从 60% 降到 38%）
- [x] 90 组参数网格优化
- [x] 部署到 run_live.py：SMA200 全局过滤 + VIX 分级 + 双动量轮换
- [x] 删除 3 个失败策略文件 + 3 个过时脚本
- [x] 清空所有日内策略代码
- [x] Memory-bank 更新

## 待开发

### 策略深化
- [ ] 探索更长持仓周期策略（周级别/月级别）
- [ ] 跨资产相关性分析
- [ ] 期权交易支持（账户增长到 $10,000+ 后考虑）

### AI/ML 增强
- [ ] 机器学习预测模型
- [ ] 自适应参数调优
- [ ] 智能仓位管理

### 基础设施改进
- [ ] Telegram Bot 通知
- [ ] SQLite 交易记录持久化
- [ ] Web 仪表盘
- [ ] CI/CD 自动测试

## 已知问题
- FutuOpenD 未安装时系统以 dry-run 模式运行
- Telegram 通知默认关闭
- 杠杆 ETF 日内 alpha 极难获取（8 年真实数据验证全部失败）
- 宏观趋势策略 Sharpe 天花板约 0.8（无法独立达到 1.0+）
