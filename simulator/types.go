package main

import (
	"fmt"
	"time"
)

// Control message types
type ControlMessage struct {
	Type         string          `json:"type"` // "control"
	SessionID    string          `json:"session_id"`
	Event        string          `json:"event"` // "start", "pause", "stop"
	TimestampUTC string          `json:"timestamp_utc"`
	Simulator    *SimulatorInfo  `json:"simulator,omitempty"`
	Race         *RaceInfo       `json:"race,omitempty"`
}

type SimulatorInfo struct {
	Source         string  `json:"source"`          // "csv" or "synthetic"
	TrackID        string  `json:"track_id"`
	PlaybackSpeed  float64 `json:"playback_speed"`
	SampleRateHz   int     `json:"sample_rate_hz"`
}

type RaceInfo struct {
	Series      string `json:"series"`
	Season      int    `json:"season"`
	EventName   string `json:"event_name"`
	SessionType string `json:"session_type"` // "practice", "qualifying", "race"
}

// Telemetry message types
type TelemetryMessage struct {
	Type           string             `json:"type"` // "telemetry"
	SessionID      string             `json:"session_id"`
	CarID          string             `json:"car_id"`
	TrackID        string             `json:"track_id"`
	MessageID      string             `json:"message_id"`
	Sequence       int64              `json:"sequence"`
	TimestampUTC   string             `json:"timestamp_utc"`
	ECUTimestampMs int64              `json:"ecu_timestamp_ms"`
	MetaTimeMs     int64              `json:"meta_time_ms"`
	Position       TelemetryPosition  `json:"position"`
	Dynamics       TelemetryDynamics  `json:"dynamics"`
	Environment    *TelemetryEnvironment `json:"environment,omitempty"`
}

type TelemetryPosition struct {
	Lap          int     `json:"lap"`
	LapDistanceM float64 `json:"lap_distance_m"`
	TrackLengthM float64 `json:"track_length_m"`
	SectionID    string  `json:"section_id"`
	SectorID     string  `json:"sector_id"`
	GPSLat       float64 `json:"gps_lat,omitempty"`
	GPSLon       float64 `json:"gps_lon,omitempty"`
}

type TelemetryDynamics struct {
	SpeedKmh              float64 `json:"speed_kmh"`
	Gear                  int     `json:"gear"`
	RPM                   int     `json:"rpm"`
	ThrottleBladePct      float64 `json:"throttle_blade_pct"`
	ThrottlePedalPct      float64 `json:"throttle_pedal_pct"`
	BrakePressureFrontBar float64 `json:"brake_pressure_front_bar"`
	BrakePressureRearBar  float64 `json:"brake_pressure_rear_bar"`
	AccelLongG            float64 `json:"accel_long_g"`
	AccelLatG             float64 `json:"accel_lat_g"`
	SteeringAngleDeg      float64 `json:"steering_angle_deg"`
}

type TelemetryEnvironment struct {
	AirTempC        float64 `json:"air_temp_c"`
	TrackTempC      float64 `json:"track_temp_c"`
	HumidityPct     float64 `json:"humidity_pct"`
	WindSpeedKph    float64 `json:"wind_speed_kph"`
	WindDirectionDeg float64 `json:"wind_direction_deg"`
	Conditions      string  `json:"conditions"` // "sunny", "cloudy", "rain"
}

// CSV row structure (maps to your telemetry CSV columns)
type CSVTelemetryRow struct {
	Timestamp              string
	MetaTime               string
	VehicleNumber          string
	Lap                    int
	LaptriggerLapdistDls   float64
	Speed                  float64
	Gear                   int
	Nmot                   int
	Ath                    float64
	Aps                    float64
	PbrakF                 float64
	PbrakR                 float64
	AccxCan                float64
	AccyCan                float64
	SteeringAngle          float64
}

// Helper to convert CSV row to TelemetryMessage
func (row *CSVTelemetryRow) ToTelemetryMessage(sessionID, trackID string, sequence int64) *TelemetryMessage {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	
	return &TelemetryMessage{
		Type:           "telemetry",
		SessionID:      sessionID,
		CarID:          "GR86-" + row.VehicleNumber,
		TrackID:        trackID,
		MessageID:      generateMessageID(sessionID, row.VehicleNumber, sequence),
		Sequence:       sequence,
		TimestampUTC:   now,
		ECUTimestampMs: parseTimestamp(row.Timestamp),
		MetaTimeMs:     parseTimestamp(row.MetaTime),
		Position: TelemetryPosition{
			Lap:          row.Lap,
			LapDistanceM: row.LaptriggerLapdistDls,
			TrackLengthM: 4200.0, // Default, update based on track
			SectionID:    determineSectionID(row.LaptriggerLapdistDls),
			SectorID:     determineSectorID(row.Lap, row.LaptriggerLapdistDls),
		},
		Dynamics: TelemetryDynamics{
			SpeedKmh:              row.Speed,
			Gear:                  row.Gear,
			RPM:                   row.Nmot,
			ThrottleBladePct:      row.Ath,
			ThrottlePedalPct:      row.Aps,
			BrakePressureFrontBar: row.PbrakF,
			BrakePressureRearBar:  row.PbrakR,
			AccelLongG:            row.AccxCan,
			AccelLatG:             row.AccyCan,
			SteeringAngleDeg:      row.SteeringAngle,
		},
	}
}

func generateMessageID(sessionID, carID string, sequence int64) string {
	return fmt.Sprintf("msg_%s_%s_%04d", sessionID, carID, sequence)
}

func parseTimestamp(ts string) int64 {
	// Parse timestamp string and convert to milliseconds
	// Implementation depends on your timestamp format
	return 0 // Placeholder
}

func determineSectionID(lapDistM float64) string {
	// Map lap distance to track sections (S1.a, S1.b, S2.a, etc.)
	// This should be based on actual track map data
	if lapDistM < 1000 {
		return "S1.a"
	} else if lapDistM < 2000 {
		return "S1.b"
	} else if lapDistM < 3000 {
		return "S2.a"
	} else {
		return "S2.b"
	}
}

func determineSectorID(lap int, lapDistM float64) string {
	// Map lap distance to sectors (typically 3 sectors per lap)
	progress := lapDistM / 4200.0 // Assume 4200m track length
	if progress < 0.33 {
		return "sector_1"
	} else if progress < 0.66 {
		return "sector_2"
	} else {
		return "sector_3"
	}
}
