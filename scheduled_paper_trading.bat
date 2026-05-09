@echo off
REM FUTU-QUANT Paper Trading Daily Update
REM 1. Record daily NAV for all models
REM 2. Auto-rebalance on 1st of month
REM Schedule: daily 03:30 (ET 16:30, after factor update)

cd /d "c:\Users\hlin2\FUTU-QUANT"

echo [%date% %time%] Paper trading daily started >> data_store\logs\scheduler.log

"C:\Users\hlin2\AppData\Local\Programs\Python\Python310\python.exe" run_paper_trading.py --daily >> data_store\logs\scheduler.log 2>&1

echo [%date% %time%] Paper trading daily finished >> data_store\logs\scheduler.log
