@echo off
REM FUTU-QUANT Factor Library Daily Update
REM 1. Incremental price update (yfinance)
REM 2. Recompute all factors
REM Schedule: daily 03:15 (ET 16:15, after market close)

cd /d "c:\Users\hlin2\FUTU-QUANT"

echo [%date% %time%] Factor library update started >> data_store\logs\scheduler.log

"C:\Users\hlin2\AppData\Local\Programs\Python\Python310\python.exe" run_factor_library.py --update >> data_store\logs\scheduler.log 2>&1
"C:\Users\hlin2\AppData\Local\Programs\Python\Python310\python.exe" run_factor_library.py --compute-factors >> data_store\logs\scheduler.log 2>&1
"C:\Users\hlin2\AppData\Local\Programs\Python\Python310\python.exe" run_factor_library.py --train-regime >> data_store\logs\scheduler.log 2>&1

echo [%date% %time%] Factor library update finished >> data_store\logs\scheduler.log
