package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	// Parse command line flags
	mode := flag.String("mode", "csv", "Simulator mode: csv or synthetic")
	csvPath := flag.String("csv", "", "Path to CSV file (required for csv mode)")
	trackID := flag.String("track", "barber", "Track ID (barber or cota)")
	speed := flag.Float64("speed", 1.0, "Playback speed multiplier (0.5-4.0)")
	sampleRate := flag.Int("rate", 10, "Sample rate in Hz")
	backendURL := flag.String("url", "ws://localhost:8080/telemetry/stream", "Backend WebSocket URL")
	
	flag.Parse()

	// Validate inputs
	if *mode == "csv" && *csvPath == "" {
		log.Fatal("CSV path is required for csv mode. Use -csv flag")
	}

	config := SimulatorConfig{
		Mode:          *mode,
		CSVPath:       *csvPath,
		TrackID:       *trackID,
		PlaybackSpeed: *speed,
		SampleRateHz:  *sampleRate,
		BackendURL:    *backendURL,
		BasePace:      120.0,
		Variability:   0.1,
	}

	// Create simulator
	sim := NewSimulator(config)

	// Connect to backend
	log.Println("Connecting to backend...")
	err := sim.Connect()
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer sim.Disconnect()

	// Start simulator
	log.Println("Starting simulator...")
	err = sim.Start()
	if err != nil {
		log.Fatalf("Failed to start: %v", err)
	}

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	<-sigChan
	log.Println("\nShutdown signal received")
	sim.Stop()
	log.Println("Simulator stopped")
}
