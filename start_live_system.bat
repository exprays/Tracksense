@echo off
REM Start Live Telemetry System - All Components
REM This script starts the receiver, dashboard, and simulator in separate windows

echo ========================================
echo Toyota GR Cup Live Telemetry System
echo ========================================
echo.
echo Starting components in separate windows...
echo.

REM Start Python Receiver
echo [1/3] Starting Python WebSocket Receiver...
start "Telemetry Receiver" cmd /k "python -m src.realtime.receiver"
timeout /t 3 /nobreak >nul

REM Start Streamlit Dashboard
echo [2/3] Starting Streamlit Dashboard...
start "Live Dashboard" cmd /k "streamlit run app_live.py"
timeout /t 3 /nobreak >nul

REM Start Go Simulator
echo [3/3] Starting Go Simulator...
echo.
echo Select mode:
echo   1. CSV Playback (Barber Race 1)
echo   2. CSV Playback (COTA Race 1)
echo   3. Synthetic Mode
echo.
set /p mode="Enter choice (1-3): "

if "%mode%"=="1" (
    start "Go Simulator" cmd /k "cd simulator && go run . -mode csv -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv -track barber -speed 1.0 -rate 10"
) else if "%mode%"=="2" (
    start "Go Simulator" cmd /k "cd simulator && go run . -mode csv -csv ../dataset/circuit-of-the-americas/COTA/Race 1/R1_cota_telemetry_data.csv -track cota -speed 1.0 -rate 10"
) else if "%mode%"=="3" (
    start "Go Simulator" cmd /k "cd simulator && go run . -mode synthetic -track barber -rate 10"
) else (
    echo Invalid choice. Starting synthetic mode...
    start "Go Simulator" cmd /k "cd simulator && go run . -mode synthetic -track barber -rate 10"
)

echo.
echo.
echo ========================================
echo All components started!
echo ========================================
echo.
echo Windows opened:
echo   - Telemetry Receiver (Port 8080)
echo   - Live Dashboard (Port 8501)
echo   - Go Simulator
echo.
echo Dashboard URL: http://localhost:8501
echo Receiver: ws://localhost:8080
echo.
echo To stop: Close all windows or press Ctrl+C in each
echo.
pause
