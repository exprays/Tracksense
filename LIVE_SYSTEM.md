# Live Telemetry System - Quick Start Guide

## Architecture Overview

```
┌──────────────┐      WebSocket       ┌──────────────┐      Buffer      ┌──────────────┐
│              │ ──────────────────▶  │              │ ───────────────▶  │              │
│ Go Simulator │   JSON Messages      │   Python     │   In-Memory      │  Streamlit   │
│              │                       │   Receiver   │   Storage        │  Dashboard   │
│              │ ◀──────────────────  │              │ ◀───────────────  │              │
└──────────────┘    Control Msgs      └──────────────┘    Read Data     └──────────────┘
       │                                      │                                 │
       │                                      │                                 │
   CSV Files                            ML Models                        Live Charts
   Telemetry                          Feature Engine                    Predictions
```

## System Components

### 1. Go Simulator (`simulator/`)

- Reads CSV telemetry files or generates synthetic data
- Streams telemetry over WebSocket at configurable rate
- Supports playback speed control (0.5x - 4x)
- Session management (start/pause/stop)

### 2. Python Receiver (`src/realtime/receiver.py`)

- WebSocket server listening on `ws://localhost:8080/telemetry/stream`
- Validates incoming messages
- Stores in thread-safe buffer
- Triggers ML feature generation and predictions

### 3. Streamlit Live Dashboard (`app_live.py`)

- Real-time visualization of telemetry
- Live gauges for speed, RPM, throttle, brake
- ML predictions: lap time, tire deg, overtake probability, incident risk
- Strategy alerts and recommendations
- Auto-refresh at configurable rate (1-10 Hz)

## Installation

### Prerequisites

- Python 3.9+
- Go 1.21+

### Setup

1. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

2. **Install Go dependencies:**

```bash
cd simulator
go mod tidy
cd ..
```

## Usage

### Step 1: Start Python Receiver

Open terminal 1:

```bash
python -m src.realtime.receiver
```

Expected output:

```
INFO:websockets.server:server listening on 0.0.0.0:8080
INFO:__main__:Telemetry receiver started on ws://0.0.0.0:8080
INFO:__main__:Waiting for simulator connections...
```

### Step 2: Start Live Dashboard

Open terminal 2:

```bash
streamlit run app_live.py
```

Dashboard will open at `http://localhost:8501`

### Step 3: Start Simulator

Open terminal 3:

**CSV Playback Mode:**

```bash
cd simulator
go run . -mode csv -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv -track barber -speed 1.0 -rate 10
```

**Synthetic Mode:**

```bash
cd simulator
go run . -mode synthetic -track barber -rate 10
```

## Configuration Options

### Simulator Flags

| Flag     | Default                                | Description                              |
| -------- | -------------------------------------- | ---------------------------------------- |
| `-mode`  | `csv`                                  | Mode: `csv` or `synthetic`               |
| `-csv`   | -                                      | Path to CSV file (required for csv mode) |
| `-track` | `barber`                               | Track ID: `barber` or `cota`             |
| `-speed` | `1.0`                                  | Playback speed (0.5-4.0)                 |
| `-rate`  | `10`                                   | Sample rate in Hz                        |
| `-url`   | `ws://localhost:8080/telemetry/stream` | Backend URL                              |

### Dashboard Settings

- **Auto-refresh**: Toggle in sidebar
- **Refresh rate**: 1-10 Hz (sidebar slider)
- **Session selection**: Dropdown in sidebar
- **Car selection**: Dropdown in sidebar

## Example Workflows

### Workflow 1: Replay Barber Race 1 at 2x Speed

```bash
# Terminal 1
python -m src.realtime.receiver

# Terminal 2
streamlit run app_live.py

# Terminal 3
cd simulator
go run . -mode csv \
  -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv \
  -track barber \
  -speed 2.0 \
  -rate 20
```

### Workflow 2: Generate Synthetic Data for Testing

```bash
# Terminal 1
python -m src.realtime.receiver

# Terminal 2
streamlit run app_live.py

# Terminal 3
cd simulator
go run . -mode synthetic -track cota -rate 10
```

### Workflow 3: High-Speed Playback for Load Testing

```bash
# Terminal 1
python -m src.realtime.receiver

# Terminal 2
streamlit run app_live.py

# Terminal 3
cd simulator
go run . -mode csv \
  -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv \
  -track barber \
  -speed 4.0 \
  -rate 50
```

## Message Protocol

### Control Message (Session Start)

```json
{
  "type": "control",
  "session_id": "session_2025-01-19_143022",
  "event": "start",
  "timestamp_utc": "2025-01-19T14:30:22.123Z",
  "simulator": {
    "source": "csv",
    "track_id": "barber",
    "playback_speed": 1.0,
    "sample_rate_hz": 10
  },
  "race": {
    "series": "TGRNA_GR_CUP_NA",
    "season": 2025,
    "event_name": "Live Simulation",
    "session_type": "race"
  }
}
```

### Telemetry Message

```json
{
  "type": "telemetry",
  "session_id": "session_2025-01-19_143022",
  "car_id": "GR86-123",
  "track_id": "barber",
  "message_id": "msg_session_2025-01-19_143022_GR86-123_0001",
  "sequence": 1,
  "timestamp_utc": "2025-01-19T14:30:22.223Z",
  "ecu_timestamp_ms": 1737295822223,
  "meta_time_ms": 1737295822223,
  "position": {
    "lap": 5,
    "lap_distance_m": 1234.5,
    "track_length_m": 4200.0,
    "section_id": "s12",
    "sector_id": "sector_2"
  },
  "dynamics": {
    "speed_kmh": 187.3,
    "gear": 5,
    "rpm": 6432,
    "throttle_blade_pct": 87.2,
    "throttle_pedal_pct": 89.1,
    "brake_pressure_front_bar": 0.0,
    "brake_pressure_rear_bar": 0.0,
    "accel_long_g": 0.52,
    "accel_lat_g": 1.23,
    "steering_angle_deg": -12.5
  }
}
```

## ML Predictions

The system generates real-time predictions:

1. **Lap Time Prediction**: Estimated lap completion time with confidence interval
2. **Tire Degradation**: Percentage degradation based on driving style
3. **Overtake Probability**: Likelihood of overtaking in next 3 laps
4. **Incident Risk**: Risk score (low/medium/high) based on driving aggression
5. **Strategy Alerts**: Contextual recommendations (pit stops, pace management)

## Troubleshooting

### "Connection refused" Error

Ensure Python receiver is running:

```bash
python -m src.realtime.receiver
```

Check that port 8080 is not blocked by firewall.

### No Data in Dashboard

1. Check receiver is running in terminal 1
2. Check simulator is running in terminal 3
3. Verify WebSocket connection in receiver logs (look for "Client connected")
4. Check `live_data/` folder for session JSON files
5. Check session selection in dashboard sidebar
6. Try refreshing the dashboard browser page

### Slow Performance

1. Reduce simulator sample rate: `-rate 5`
2. Lower dashboard refresh rate (sidebar slider)
3. Reduce playback speed: `-speed 0.5`

### CSV File Not Found

Use correct relative path from simulator directory:

```bash
cd simulator
go run . -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv
```

## Performance Considerations

- **Memory**: Buffer stores last 1000 messages per car
- **CPU**: Feature generation runs async to avoid blocking
- **Network**: WebSocket messages are ~500 bytes each
- **Latency**: Typical end-to-end latency: 50-100ms at 10Hz

## Next Steps

- [ ] Add Redis for distributed buffer (multi-instance support)
- [ ] Implement actual ML models (replace placeholder predictions)
- [ ] Add multi-car comparison view
- [ ] Create simulator control panel UI
- [ ] Add historical playback controls
- [ ] Integrate weather data streaming
