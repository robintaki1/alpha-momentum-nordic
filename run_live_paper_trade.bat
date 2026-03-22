@echo off
setlocal

cd /d "%~dp0"
python "python\live_paper_engine.py" --capital-sek 50000

endlocal
