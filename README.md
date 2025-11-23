# Real-Time Race Strategy Optimizer

A comprehensive real-time analytics and strategy tool for the Toyota GR Cup Series that helps teams make optimal pit stop decisions, predict tire degradation, and maximize race performance using advanced machine learning and AI-powered insights.

## 🔴 NEW: Live Telemetry System

**Real-time race analytics with streaming telemetry data!**

- 🏎️ **Go Simulator**: Stream telemetry from CSV files or generate synthetic data
- 📡 **WebSocket Streaming**: Real-time data transmission at 10+ Hz
- 🎛️ **Live Dashboard**: Auto-refreshing gauges, charts, and ML predictions
- 🤖 **Real-time ML**: Instant predictions for lap times, tire deg, overtake probability
- 🎯 **Strategy Alerts**: Live recommendations and incident risk warnings

**[→ Live System Quick Start Guide](LIVE_SYSTEM.md)**

```bash
# Start all components (receiver, dashboard, simulator)
./start_live_system.bat   # Windows
./start_live_system.sh    # Linux/Mac
```

## 🏁 Features

### Core Analytics

- **ML-Powered Tire Degradation**: Gradient Boosting model trained on historical race data predicts tire wear with high accuracy
- **Intelligent Pit Strategy**: XGBoost classifier recommends optimal pit stop timing based on tire life, fuel, and race conditions
- **Fuel Consumption Monitor**: Real-time fuel tracking with lap-by-lap predictions
- **Weather-Aware Strategy**: Integrated weather data for strategy adjustments
- **Race Simulator**: Test and compare multiple race strategies

### Advanced ML Models

- **Tire Degradation Model**: Predicts tire life based on lap data, weather, and driving style (R² > 0.85)
- **Pit Strategy Optimizer**: Machine learning model that identifies optimal pit windows (F1 Score > 0.80)
- **Driver Fingerprinting**: Classifies and analyzes unique driving characteristics per driver
- **Predictive Analytics**: Forecasts race outcomes and identifies risk factors

### AI-Powered Insights

- **Natural Language Insights**: AI-generated recommendations in plain English
- **Priority Action System**: Intelligent urgency-based recommendation ranking
- **Performance Analysis**: Automated driver performance evaluation with improvement suggestions
- **Race Outcome Predictions**: Probability-based finish predictions with success metrics

### Interactive Dashboard

- **7 Specialized Tabs**: Overview, Pit Strategy, Performance, Fuel, Conditions, AI Insights, Model Training
- **Real-time Visualizations**: 12+ interactive Plotly charts
- **Lap-by-Lap Replay**: Simulate any point in the race
- **Model Training Interface**: Train and evaluate ML models directly in the dashboard

## 📊 Data Sources

- Telemetry data (speed, throttle, brake, steering)
- Lap timing data (lap times, sector times)
- Race results and positions
- Weather conditions
- Driver performance metrics

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train ML Models (Recommended First Step)

```bash
# Train all models on historical race data
python train_models.py
```

This will train:

- Tire degradation prediction model
- Pit strategy classifier
- Driver fingerprinting model

Training takes 1-2 minutes and creates model files in the `models/` directory.

### Run the Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Run Tests

```bash
# Test core components
python tests\test_components.py

# Test ML models and AI insights
python tests\test_ml_models.py
```

## 📁 Project Structure

```
toyota/
├── app.py                          # Main Streamlit dashboard
├── train_models.py                 # Standalone model training script
├── src/
│   ├── data/
│   │   ├── loader.py              # Data loading utilities
│   │   ├── preprocessor.py        # Feature engineering (67 features)
│   │   └── visualizer.py          # Plotly chart generators
│   ├── models/
│   │   ├── tire_model.py          # Gradient Boosting tire predictor
│   │   ├── fuel_model.py          # Fuel consumption calculator
│   │   ├── pit_optimizer.py       # Pit stop strategy optimizer
│   │   ├── model_trainer.py       # ML training pipeline
│   │   └── ai_insights.py         # AI-powered insights generator
│   ├── analytics/
│   │   └── race_simulator.py      # Race scenario simulation
│   │   ├── race_simulator.py  # Race strategy simulator
│   │   └── weather_impact.py  # Weather analysis
│   └── utils/
│       ├── constants.py       # Configuration constants
│       └── helpers.py         # Helper functions
├── dataset/                    # Race data files
├── models/                     # Saved ML models
└── requirements.txt
```

## 🎯 Hackathon Category

**Real-Time Analytics** - Design a tool that simulates real-time decision-making for a race engineer.

## 📝 License

MIT License
