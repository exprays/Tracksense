# Real-Time Race Strategy Optimizer

A comprehensive real-time analytics and strategy tool for the GR Cup Series that helps teams make optimal pit stop decisions, predict tire degradation, and maximize race performance.

## 🏁 Features

- **Live Tire Degradation Prediction**: Machine learning model that predicts tire wear based on telemetry data
- **Optimal Pit Window Calculator**: Real-time calculation of the best pit stop windows
- **Fuel Consumption Monitor**: Track fuel usage and predict remaining laps
- **Weather-Aware Strategy**: Integrate weather data for strategy adjustments
- **Race Simulator**: Test different strategies against historical race data
- **Interactive Dashboard**: Real-time visualization of all key metrics

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

### Run the Dashboard

```bash
streamlit run app.py
```

## 📁 Project Structure

```
toyota/
├── app.py                      # Main Streamlit dashboard
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessor.py    # Data preprocessing
│   ├── models/
│   │   ├── tire_model.py      # Tire degradation prediction
│   │   ├── fuel_model.py      # Fuel consumption prediction
│   │   └── pit_optimizer.py   # Pit stop optimization
│   ├── analytics/
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
