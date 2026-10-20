@echo off
setlocal
cd /d "%~dp0"
if exist "dist\MouseLimbLab\MouseLimbLab.exe" (
  start "" "dist\MouseLimbLab\MouseLimbLab.exe"
  exit /b 0
)
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto missing
)
if not exist ".venv\mouselimblab_ready" (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  echo ready>".venv\mouselimblab_ready"
)
".venv\Scripts\python.exe" launch.py
if errorlevel 1 goto fail
exit /b 0
:missing
echo Install Python 3.12 64-bit from https://www.python.org/downloads/windows/
echo Then double-click Start_Windows.bat again.
pause
exit /b 1
:fail
echo Launch failed. The error is shown above.
pause
exit /b 1
