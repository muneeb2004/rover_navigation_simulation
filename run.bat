@echo off
setlocal

python "%~dp0rover_navigation_simulation.py" %*

if errorlevel 1 (
    echo.
    echo Program exited with an error.
    pause
    exit /b %errorlevel%
)

if "%~1"=="" (
    echo.
    echo Command completed. Press any key to exit...
    pause >nul
)

