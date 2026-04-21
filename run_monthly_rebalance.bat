@echo off
setlocal

cd /d "%~dp0"
python "python\monthly_rebalance_runner.py" --capital-sek 50000

endlocal
