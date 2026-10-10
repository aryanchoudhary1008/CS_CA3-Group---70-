@echo off
REM Windows: creates a virtual environment, installs requirements and runs the project.
REM Usage: run_project.bat            (full run, 20 seeds)
REM        run_project.bat --seeds 3  (quick demo)
cd /d %~dp0
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\zero_trust_prototype.py %*
echo.
echo Results are in the folder: outputs_ca2_v2
pause
