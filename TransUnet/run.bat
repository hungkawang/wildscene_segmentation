@echo off
REM Check if main.py exists in the current directory
if exist main.py (
    REM Run main.py
    python main.py
) else (
    echo main.py does not exist!
)

REM Wait for the user to press any key before closing the window
pause