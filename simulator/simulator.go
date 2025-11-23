package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/gorilla/websocket"
)

type Simulator struct {
	sessionID     string
	conn          *websocket.Conn
	config        SimulatorConfig
	isRunning     bool
	isPaused      bool
	sequence      int64
	stopChan      chan bool
}

type SimulatorConfig struct {
	Mode           string  // "csv" or "synthetic"
	CSVPath        string
	TrackID        string
	PlaybackSpeed  float64
	SampleRateHz   int
	BackendURL     string
	DriverProfile  string  // For synthetic mode
	BasePace       float64
	Variability    float64
	MistakeFreq    float64
}

func NewSimulator(config SimulatorConfig) *Simulator {
	sessionID := "session_" + time.Now().Format("2006-01-02_150405")
	
	return &Simulator{
		sessionID: sessionID,
		config:    config,
		isRunning: false,
		isPaused:  false,
		sequence:  0,
		stopChan:  make(chan bool),
	}
}

func (s *Simulator) Connect() error {
	var err error
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
		ReadBufferSize:   4096,
		WriteBufferSize:  4096,
	}
	s.conn, _, err = dialer.Dial(s.config.BackendURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to backend: %w", err)
	}
	
	// Set read/write deadlines to prevent timeout
	s.conn.SetReadDeadline(time.Time{})
	s.conn.SetWriteDeadline(time.Time{})
	
	// Enable pong handler for keepalive
	s.conn.SetPongHandler(func(string) error {
		s.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})
	
	log.Printf("Connected to backend: %s", s.config.BackendURL)
	return nil
}

func (s *Simulator) Disconnect() error {
	if s.conn != nil {
		return s.conn.Close()
	}
	return nil
}

func (s *Simulator) Start() error {
	if s.isRunning {
		return fmt.Errorf("simulator already running")
	}

	// Send start control message
	err := s.sendControlMessage("start")
	if err != nil {
		return err
	}

	s.isRunning = true
	log.Printf("Simulator started - Session: %s", s.sessionID)

	// Start streaming based on mode
	if s.config.Mode == "csv" {
		go s.streamCSV()
	} else {
		go s.streamSynthetic()
	}

	return nil
}

func (s *Simulator) Pause() error {
	if !s.isRunning {
		return fmt.Errorf("simulator not running")
	}
	s.isPaused = !s.isPaused
	
	event := "pause"
	if !s.isPaused {
		event = "resume"
	}
	
	return s.sendControlMessage(event)
}

func (s *Simulator) Stop() error {
	if !s.isRunning {
		return fmt.Errorf("simulator not running")
	}

	s.stopChan <- true
	s.isRunning = false
	
	return s.sendControlMessage("stop")
}

func (s *Simulator) sendControlMessage(event string) error {
	msg := ControlMessage{
		Type:         "control",
		SessionID:    s.sessionID,
		Event:        event,
		TimestampUTC: time.Now().UTC().Format(time.RFC3339Nano),
	}

	if event == "start" {
		msg.Simulator = &SimulatorInfo{
			Source:        s.config.Mode,
			TrackID:       s.config.TrackID,
			PlaybackSpeed: s.config.PlaybackSpeed,
			SampleRateHz:  s.config.SampleRateHz,
		}
		msg.Race = &RaceInfo{
			Series:      "TGRNA_GR_CUP_NA",
			Season:      2025,
			EventName:   "Live Simulation",
			SessionType: "race",
		}
	}

	return s.conn.WriteJSON(msg)
}

func (s *Simulator) streamCSV() error {
	file, err := os.Open(s.config.CSVPath)
	if err != nil {
		return fmt.Errorf("failed to open CSV: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	
	// Read header
	header, err := reader.Read()
	if err != nil {
		return err
	}
	
	colMap := mapCSVColumns(header)
	log.Printf("CSV columns mapped: %d columns found", len(colMap))

	// Calculate delay between messages
	intervalMs := int(1000.0 / float64(s.config.SampleRateHz) / s.config.PlaybackSpeed)
	ticker := time.NewTicker(time.Duration(intervalMs) * time.Millisecond)
	defer ticker.Stop()

	rowCount := 0
	for {
		select {
		case <-s.stopChan:
			log.Println("Stopping CSV stream")
			return nil
		case <-ticker.C:
			if s.isPaused {
				continue
			}

			record, err := reader.Read()
			if err == io.EOF {
				log.Println("CSV stream complete")
				s.Stop()
				return nil
			}
			if err != nil {
				log.Printf("Error reading CSV row: %v", err)
				continue
			}

			// Parse row and send telemetry
			row := parseCSVRow(record, colMap)
			if row != nil {
				s.sequence++
				msg := row.ToTelemetryMessage(s.sessionID, s.config.TrackID, s.sequence)
				
			s.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			err = s.conn.WriteJSON(msg)
			if err != nil {
				log.Printf("Error sending telemetry: %v", err)
				// Don't spam errors - stop on write failure
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Println("Connection lost, stopping stream")
					s.Stop()
					return err
				}
				continue
			}
			
			rowCount++
			if rowCount%100 == 0 {
				log.Printf("Sent %d telemetry messages", rowCount)
			}
			}
		}
	}
}

func (s *Simulator) streamSynthetic() error {
	// Synthetic data generation
	intervalMs := int(1000.0 / float64(s.config.SampleRateHz))
	ticker := time.NewTicker(time.Duration(intervalMs) * time.Millisecond)
	defer ticker.Stop()

	lap := 1
	lapDistance := 0.0
	trackLength := 4200.0 // meters
	baseSpeed := 120.0    // km/h

	log.Println("Starting synthetic telemetry stream")

	for {
		select {
		case <-s.stopChan:
			log.Println("Stopping synthetic stream")
			return nil
		case <-ticker.C:
			if s.isPaused {
				continue
			}

			s.sequence++
			
			// Generate synthetic telemetry
			msg := s.generateSyntheticTelemetry(lap, lapDistance, baseSpeed)
			
			s.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			err := s.conn.WriteJSON(msg)
			if err != nil {
				log.Printf("Error sending synthetic telemetry: %v", err)
				// Stop on connection error
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Println("Connection lost, stopping stream")
					s.Stop()
					return err
				}
				continue
			}

			// Update lap distance
			lapDistance += baseSpeed * (float64(intervalMs) / 1000.0) / 3.6 // km/h to m/s
			if lapDistance >= trackLength {
				lap++
				lapDistance = 0
				log.Printf("Completed lap %d", lap-1)
			}

			if s.sequence%100 == 0 {
				log.Printf("Sent %d synthetic messages", s.sequence)
			}
		}
	}
}

func (s *Simulator) generateSyntheticTelemetry(lap int, lapDistance, baseSpeed float64) *TelemetryMessage {
	now := time.Now().UTC()
	
	// Add variability
	speed := baseSpeed + (rand.Float64()-0.5)*s.config.Variability*baseSpeed
	rpm := int(4000 + speed*30)
	throttle := 50.0 + (rand.Float64()-0.5)*40
	
	// Simulate braking zones (roughly every 1000m)
	lapProgress := lapDistance / 4200.0
	isBrakingZone := (int(lapProgress*4) % 2) == 1 // Brake in zones 25-50% and 75-100%
	
	brakeFront := 0.0
	brakeRear := 0.0
	accelLong := 0.5
	
	if isBrakingZone && rand.Float64() > 0.3 {
		brakeFront = 10.0 + rand.Float64()*30.0 // 10-40 bar
		brakeRear = brakeFront * 0.7            // Rear is ~70% of front
		accelLong = -0.8 - rand.Float64()*0.5   // Negative G during braking
		throttle = 0.0                          // No throttle when braking
	}
	
	return &TelemetryMessage{
		Type:           "telemetry",
		SessionID:      s.sessionID,
		CarID:          "GR86-SYNTHETIC-001",
		TrackID:        s.config.TrackID,
		MessageID:      generateMessageID(s.sessionID, "001", s.sequence),
		Sequence:       s.sequence,
		TimestampUTC:   now.Format(time.RFC3339Nano),
		ECUTimestampMs: now.UnixMilli(),
		MetaTimeMs:     now.UnixMilli(),
		Position: TelemetryPosition{
			Lap:          lap,
			LapDistanceM: lapDistance,
			TrackLengthM: 4200.0,
			SectionID:    determineSectionID(lapDistance),
			SectorID:     determineSectorID(lap, lapDistance),
		},
		Dynamics: TelemetryDynamics{
			SpeedKmh:              speed,
			Gear:                  int(speed/40) + 1,
			RPM:                   rpm,
			ThrottleBladePct:      throttle,
			ThrottlePedalPct:      throttle,
			BrakePressureFrontBar: brakeFront,
			BrakePressureRearBar:  brakeRear,
			AccelLongG:            accelLong,
			AccelLatG:             0.8 + (rand.Float64()-0.5)*0.4,
			SteeringAngleDeg:      (rand.Float64() - 0.5) * 180,
		},
	}
}

func mapCSVColumns(header []string) map[string]int {
	colMap := make(map[string]int)
	for i, col := range header {
		colMap[col] = i
	}
	return colMap
}

func parseCSVRow(record []string, colMap map[string]int) *CSVTelemetryRow {
	row := &CSVTelemetryRow{}
	
	// Helper to safely get column value
	getCol := func(name string) string {
		if idx, ok := colMap[name]; ok && idx < len(record) {
			return record[idx]
		}
		return ""
	}
	
	getFloat := func(name string) float64 {
		val := getCol(name)
		if val == "" {
			return 0.0
		}
		f, _ := strconv.ParseFloat(val, 64)
		return f
	}
	
	getInt := func(name string) int {
		val := getCol(name)
		if val == "" {
			return 0
		}
		i, _ := strconv.Atoi(val)
		return i
	}

	row.Timestamp = getCol("timestamp")
	row.MetaTime = getCol("meta_time")
	row.VehicleNumber = getCol("vehicle_number")
	row.Lap = getInt("lap")
	row.LaptriggerLapdistDls = getFloat("laptrigger_lapdist_dls")
	row.Speed = getFloat("Speed")
	row.Gear = getInt("Gear")
	row.Nmot = getInt("nmot")
	row.Ath = getFloat("ath")
	row.Aps = getFloat("aps")
	row.PbrakF = getFloat("pbrak_f")
	row.PbrakR = getFloat("pbrak_r")
	row.AccxCan = getFloat("accx_can")
	row.AccyCan = getFloat("accy_can")
	row.SteeringAngle = getFloat("Steering_Angle")

	return row
}
