@echo off
setlocal enabledelayedexpansion
title 4-UAV Cooperative Slung Load Simulation
echo ======================================================================
echo   Cooperative 4-UAV Cable-Suspended Load Simulation Launcher
echo ======================================================================
echo.

:: Get current directory and change to it
cd /d "%~dp0"
echo Working directory: %CD%

:: Detect Python or Py launcher
set PYTHON_CMD=
where python >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
) else (
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py
    ) else (
        echo ERROR: Python was not found in your system PATH or command list.
        echo Please ensure Python is installed and "Add Python to PATH" is checked during installation.
        echo.
        pause
        exit /b 1
    )
)

echo Python command detected: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: Clear choice
set CHOICE=

echo Select simulation scenario:
echo   1. Hover (default)
echo   2. Circle trajectory
echo   3. Figure-8 (Lemniscate)
echo   4. Step response
echo   5. Run validation tests only
echo   6. Hover (no animation, plots only)
echo.
set /p CHOICE="Enter choice [1-6] (default=1): "

if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="5" (
    echo.
    echo Running test suite...
    %PYTHON_CMD% -m pytest tests/ -v
    echo.
    pause
    exit /b 0
)

set SCENARIO=hover
set EXTRA_ARGS=

if "%CHOICE%"=="2" set SCENARIO=circle
if "%CHOICE%"=="3" set SCENARIO=lemniscate
if "%CHOICE%"=="4" set SCENARIO=step
if "%CHOICE%"=="6" (
    set SCENARIO=hover
    set EXTRA_ARGS=--no-animate
)

echo.
echo Starting simulation with scenario: %SCENARIO% ...
echo.
%PYTHON_CMD% -m sim.run_sim --scenario %SCENARIO% %EXTRA_ARGS%

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Simulation crashed or failed with exit code %errorlevel%.
) else (
    echo.
    echo Simulation complete.
)

pause
