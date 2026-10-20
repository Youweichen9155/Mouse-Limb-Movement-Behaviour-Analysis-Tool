@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller==6.16.0
if errorlevel 1 goto fail
".venv\Scripts\python.exe" build_app.py
if errorlevel 1 goto fail
echo Ready: dist\MouseLimbLab\MouseLimbLab.exe
pause
exit /b 0
:fail
echo Build failed. Review the error above.
pause
exit /b 1
