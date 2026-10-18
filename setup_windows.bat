@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or 3.12 from python.org and tick "Add Python to PATH".
  pause & exit /b 1
)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (echo Failed to install PyTorch. Check your internet connection. & pause & exit /b 1)
pip install -r requirements.txt
if errorlevel 1 (echo Failed to install requirements. & pause & exit /b 1)
echo.
echo Setup finished. Next: prepare_data.bat, then train_model.bat, then run_app.bat
pause
