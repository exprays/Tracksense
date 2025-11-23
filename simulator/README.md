# Toyota Racing Simulator

Go-based telemetry simulator for streaming race data to analytics backend.

## Features

- **CSV Playback**: Stream real telemetry data from CSV files
- **Synthetic Mode**: Generate synthetic racing telemetry
- **Variable Speed**: Control playback speed (0.5x-4x)
- **WebSocket Streaming**: Real-time data transmission
- **Session Management**: Start/pause/stop controls

## Installation

```bash
cd simulator
go mod tidy
go build
```

## Usage

### CSV Playback Mode

```bash
# Stream Barber Race 1 data
./simulator -mode csv -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv -track barber -speed 1.0 -rate 10

# Stream COTA Race 2 at 2x speed
./simulator -mode csv -csv ../dataset/circuit-of-the-americas/COTA/Race\ 2/R2_cota_telemetry_data.csv -track cota -speed 2.0 -rate 20
```

### Synthetic Mode

```bash
# Generate synthetic telemetry
./simulator -mode synthetic -track barber -rate 10 -speed 1.0
```

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `-mode` | `csv` | Simulator mode: `csv` or `synthetic` |
| `-csv` | - | Path to CSV telemetry file (required for csv mode) |
| `-track` | `barber` | Track ID: `barber` or `cota` |
| `-speed` | `1.0` | Playback speed multiplier (0.5-4.0) |
| `-rate` | `10` | Sample rate in Hz |
| `-url` | `ws://localhost:8080/telemetry/stream` | Backend WebSocket URL |

## Message Protocol

### Control Messages

Sent on session start/pause/stop:

```json
{
  "type": "control",
  "session_id": "session_2025-01-19_143022",
  "event": "start",
  "timestamp_utc": "2025-01-19T14:30:22.123456Z",
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

### Telemetry Messages

Streaming telemetry data:

```json
{
  "type": "telemetry",
  "session_id": "session_2025-01-19_143022",
  "car_id": "GR86-123",
  "track_id": "barber",
  "message_id": "msg_session_2025-01-19_143022_GR86-123_0001",
  "sequence": 1,
  "timestamp_utc": "2025-01-19T14:30:22.223456Z",
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

## Architecture

```
CSV File → CSV Reader → Telemetry Message → WebSocket → Python Backend → Streamlit
                ↓                              ↓
         Playback Control            Session Management
```

## Dependencies

- `gorilla/websocket`: WebSocket client
- `google/uuid`: Unique ID generation

## Error Handling

- Automatic reconnection on connection loss
- Graceful shutdown on SIGINT/SIGTERM
- CSV parsing error recovery
- Invalid data row skipping
