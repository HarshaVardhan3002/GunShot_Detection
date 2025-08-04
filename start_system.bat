@echo off
REM Gunshot Localization System Startup Script for Windows

cd /d "C:\Users\Von3002\Desktop\Gunshot Detection\gunshot-localizer"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run the system with default configuration
python main.py --config config\default_config.json --verbose

pause
