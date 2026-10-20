@echo off
setlocal
cd /d "%~dp0"
py -3.12 --version >nul 2>&1
if errorlevel 1 goto fallback
py -3.12 install_deeplabcut.py
goto end
:fallback
python install_deeplabcut.py
:end
if errorlevel 1 echo Installation failed. Install Python 3.12 from python.org and check the log above.
pause
