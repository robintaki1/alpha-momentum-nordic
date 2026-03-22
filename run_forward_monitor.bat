@echo off
setlocal

cd /d "%~dp0"
python python\forward_monitor.py --data-dir data --results-root results --output-dir results\forward_monitor --history-months 6

endlocal
