@echo off
REM Builds a Windows executable with PyInstaller. Run this on a Windows PC (an .exe cannot be built on Linux/macOS).
cd /d %~dp0
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller --onefile --name ZeroTrustPrototype --exclude-module tkinter --collect-submodules sklearn src\zero_trust_prototype.py
echo.
echo Executable created at: dist\ZeroTrustPrototype.exe
pause
