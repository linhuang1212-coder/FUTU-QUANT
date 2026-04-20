# 技术上下文

## 运行环境
- OS: Windows 10
- Python: 3.10+
- IDE: Cursor

## 依赖
| 包 | 用途 |
|---|------|
| futu-api >= 9.1 | 富途 OpenAPI SDK |
| pandas >= 2.0 | 数据处理 |
| numpy >= 1.24 | 数值计算 |
| ta >= 0.11 | 技术分析库（备用） |
| PyYAML >= 6.0 | 配置文件解析 |
| python-telegram-bot >= 20.0 | Telegram 通知 |
| APScheduler >= 3.10 | 定时任务 |
| pytest >= 7.0 | 测试框架 |

## FutuOpenD 配置
- 网关地址: 127.0.0.1:11111
- 需要本地运行 FutuOpenD 客户端
- 支持模拟盘（SIMULATE）和实盘（REAL）

## API 限制
- futu-api 有调用频率限制
- 实时行情需要付费订阅
- 免费行情有 15 分钟延迟

## 数据存储
- 行情缓存: CSV 文件（data_store/cache/）
- 交易记录: 内存中（后续可扩展 SQLite）
- 日志: data_store/logs/futu_quant.log
