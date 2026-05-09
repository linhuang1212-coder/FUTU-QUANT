@echo off
REM FUTU-QUANT Weekly Report - Saturday 10:00 AM
REM Generates weekly performance report and sends via Telegram
REM Register: python weekly_report.py --install

cd /d "c:\Users\hlin2\FUTU-QUANT"

echo [%date% %time%] Weekly report started >> data_store\logs\scheduler.log

"C:\Users\hlin2\AppData\Local\Programs\Python\Python310\python.exe" weekly_report.py >> data_store\logs\scheduler.log 2>&1

echo [%date% %time%] Weekly report finished >> data_store\logs\scheduler.log
