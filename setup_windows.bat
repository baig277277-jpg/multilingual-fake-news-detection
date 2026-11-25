@echo off
setlocal
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Check your Python version and internet connection.
  pause
  exit /b 1
)
echo.
echo Setup complete. Run run_tests.bat, then run_demo.bat, then run_real_data.bat.
pause
