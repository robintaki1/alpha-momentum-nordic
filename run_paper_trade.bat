@echo off
setlocal
cd /d "%~dp0"
python "python\paper_trade_tracker.py" --capital-sek 50000
endlocal
