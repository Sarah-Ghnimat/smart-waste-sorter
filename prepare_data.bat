@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat
pip install -r requirements-data.txt
python prepare_dataset.py
pause
