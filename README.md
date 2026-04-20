# FUTU-QUANT

基于富途证券 OpenAPI 的美股全自动量化交易系统。

## 环境要求

- Python 3.10 或更高版本
- [FutuOpenD](https://openapi.futunn.com/futu-api-doc/opend/opend-intro.html)（富途 OpenD 网关，用于行情与交易）

## 快速开始

1. 克隆仓库后，在项目根目录安装依赖：

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. 编辑 `config/settings.yaml`（以及按需修改 `config/strategies.yaml`、`config/symbols.yaml`），填写富途连接、账户与风控等参数。

3. 确保 FutuOpenD 已启动并与配置中的 `host`、`port` 一致。

4. 运行主程序（实现后将提供 `main.py`）：

   ```powershell
   python main.py
   ```

## 项目结构概览

| 路径 | 说明 |
|------|------|
| `config/` | 运行配置：连接、风控、策略参数、交易标的 |
| `core/` | 核心编排与领域逻辑 |
| `data/` | 行情与数据访问层 |
| `strategy/` | 策略与信号生成 |
| `execution/` | 下单与订单管理 |
| `risk/` | 风控与仓位约束 |
| `notification/` | 通知（如 Telegram） |
| `backtest/` | 回测相关 |
| `utils/` | 通用工具函数 |
| `tests/` | 单元测试与集成测试 |
| `docs/` | 设计文档与规格说明 |

## 设计文档

详细设计与规格说明见仓库内目录：

[docs/superpowers/specs/](docs/superpowers/specs/)

当前设计说明文档：[2026-04-20-futu-quant-design.md](docs/superpowers/specs/2026-04-20-futu-quant-design.md)

## 免责声明

量化交易存在资金损失风险。本项目仅供学习与研究，不构成投资建议。使用本软件进行交易的一切后果由使用者自行承担。
