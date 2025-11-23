#!/bin/bash
# Start Live Telemetry System - All Components
# This script starts the receiver, dashboard, and simulator in separate terminals

echo "========================================"
echo "Toyota GR Cup Live Telemetry System"
echo "========================================"
echo ""
echo "Starting components..."
echo ""

# Check if running on Windows with Git Bash / WSL
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows environment"
    
    # Start Python Receiver
    echo "[1/3] Starting Python WebSocket Receiver..."
    start "Telemetry Receiver" python -m src.realtime.receiver &
    sleep 3
    
    # Start Streamlit Dashboard
    echo "[2/3] Starting Streamlit Dashboard..."
    start "Live Dashboard" streamlit run app_live.py &
    sleep 3
    
    # Start Go Simulator
    echo "[3/3] Starting Go Simulator..."
    echo ""
    echo "Select mode:"
    echo "  1. CSV Playback (Barber Race 1)"
    echo "  2. CSV Playback (COTA Race 1)"
    echo "  3. Synthetic Mode"
    echo ""
    read -p "Enter choice (1-3): " mode
    
    case $mode in
        1)
            cd simulator && start "Go Simulator" go run . -mode csv -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv -track barber -speed 1.0 -rate 10
            ;;
        2)
            cd simulator && start "Go Simulator" go run . -mode csv -csv "../dataset/circuit-of-the-americas/COTA/Race 1/R1_cota_telemetry_data.csv" -track cota -speed 1.0 -rate 10
            ;;
        3)
            cd simulator && start "Go Simulator" go run . -mode synthetic -track barber -rate 10
            ;;
        *)
            echo "Invalid choice. Starting synthetic mode..."
            cd simulator && start "Go Simulator" go run . -mode synthetic -track barber -rate 10
            ;;
    esac
    
else
    # Linux/Mac
    echo "Detected Unix-like environment"
    
    # Start Python Receiver
    echo "[1/3] Starting Python WebSocket Receiver..."
    gnome-terminal -- bash -c "python -m src.realtime.receiver; exec bash" 2>/dev/null || \
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && python -m src.realtime.receiver"' 2>/dev/null || \
    xterm -e "python -m src.realtime.receiver" 2>/dev/null || \
    python -m src.realtime.receiver &
    sleep 3
    
    # Start Streamlit Dashboard
    echo "[2/3] Starting Streamlit Dashboard..."
    gnome-terminal -- bash -c "streamlit run app_live.py; exec bash" 2>/dev/null || \
    xterm -e "streamlit run app_live.py" 2>/dev/null || \
    streamlit run app_live.py &
    sleep 3
    
    # Start Go Simulator
    echo "[3/3] Starting Go Simulator..."
    echo ""
    echo "Select mode:"
    echo "  1. CSV Playback (Barber Race 1)"
    echo "  2. CSV Playback (COTA Race 1)"
    echo "  3. Synthetic Mode"
    echo ""
    read -p "Enter choice (1-3): " mode
    
    case $mode in
        1)
            cd simulator && gnome-terminal -- bash -c "go run . -mode csv -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv -track barber -speed 1.0 -rate 10; exec bash" 2>/dev/null || \
            go run . -mode csv -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv -track barber -speed 1.0 -rate 10 &
            ;;
        2)
            cd simulator && gnome-terminal -- bash -c "go run . -mode csv -csv '../dataset/circuit-of-the-americas/COTA/Race 1/R1_cota_telemetry_data.csv' -track cota -speed 1.0 -rate 10; exec bash" 2>/dev/null || \
            go run . -mode csv -csv "../dataset/circuit-of-the-americas/COTA/Race 1/R1_cota_telemetry_data.csv" -track cota -speed 1.0 -rate 10 &
            ;;
        3)
            cd simulator && gnome-terminal -- bash -c "go run . -mode synthetic -track barber -rate 10; exec bash" 2>/dev/null || \
            go run . -mode synthetic -track barber -rate 10 &
            ;;
        *)
            echo "Invalid choice. Starting synthetic mode..."
            cd simulator && gnome-terminal -- bash -c "go run . -mode synthetic -track barber -rate 10; exec bash" 2>/dev/null || \
            go run . -mode synthetic -track barber -rate 10 &
            ;;
    esac
fi

echo ""
echo "========================================"
echo "All components started!"
echo "========================================"
echo ""
echo "Services:"
echo "  - Telemetry Receiver: ws://localhost:8080"
echo "  - Live Dashboard: http://localhost:8501"
echo "  - Go Simulator: Active"
echo ""
echo "Dashboard URL: http://localhost:8501"
echo "Receiver: ws://localhost:8080"
echo ""
echo "Note: Data is shared via JSON files in live_data/ folder"
echo ""
echo "To stop: Close terminals or press Ctrl+C in each"
echo ""
