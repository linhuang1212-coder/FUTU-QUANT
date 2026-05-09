@echo off
REM FUTU-QUANT Scheduled Start
REM Runs start.py --force, which waits for market open then trades
REM Requires: FutuOpenD running, Python in PATH

cd /d "c:\Users\hlin2\FUTU-QUANT"

echo [%date% %time%] FUTU-QUANT scheduled start >> data_store\logs\scheduler.log

REM 验证期: --dry-run 模式 (确认稳定后去掉 --dry-run)
"C:\Users\hlin2\AppData\Local\Programs\Python\Python310\python.exe" start.py --force --dry-run >> data_store\logs\scheduler.log 2>&1

echo [%date% %time%] FUTU-QUANT session ended >> data_store\logs\scheduler.log
