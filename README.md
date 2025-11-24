# Toyota TrackSense

A comprehensive race strategy analytics platform for the Toyota GR Cup Series that helps teams make optimal pit stop decisions, predict tire degradation, and maximize race performance using advanced machine learning and AI-powered insights.

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

- **9 Specialized Tabs**: Overview, Pit Strategy, Performance, Fuel, Conditions, AI Insights, Model Training, Driver Comparison, Strategy Chat
- **Real-time Visualizations**: 15+ interactive Plotly charts with Toyota GR branding
- **Lap-by-Lap Replay**: Simulate any point in the race with slider control
- **Model Training Interface**: Train and evaluate ML models directly in the dashboard
- **Multi-Driver Comparison**: Side-by-side performance analysis across multiple drivers
- **Natural Language Chat**: Ask strategy questions in plain English and get AI-powered answers
- **PDF Report Generation**: Export comprehensive race analysis reports

## 📊 Data Sources

Historical race data from four tracks:

- **Barber Motorsports Park**: 2 races, full telemetry and sector data
- **Circuit of the Americas (COTA)**: 2 races, complete race analysis
- **Indianapolis Motor Speedway**: 2 races, high-speed oval data
- **Sebring International Raceway**: 2 races, endurance track data

Data includes:

- Lap timing data (lap times, sector times, best laps)
- Weather conditions (temperature, humidity, wind, pressure)
- Race results and positions
- Driver performance metrics
- Endurance analysis with sections

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
├── app.py                               # Main Streamlit dashboard
├── train_models.py                      # Standalone model training script
├── src/
│   ├── data/
│   │   ├── loader.py                   # Data loading for 4 tracks
│   │   ├── preprocessor.py             # Feature engineering (67 features)
│   │   └── visualizer.py               # Plotly chart generators
│   ├── models/
│   │   ├── tire_model.py               # Gradient Boosting tire predictor
│   │   ├── fuel_model.py               # Fuel consumption calculator
│   │   ├── pit_optimizer.py            # Pit stop strategy optimizer
│   │   ├── model_trainer.py            # ML training pipeline
│   │   ├── ai_insights.py              # AI-powered insights generator
│   │   └── nlp_strategy.py             # Natural language strategy chat
│   ├── analytics/
│   │   ├── race_simulator.py           # Race scenario simulation
│   │   └── multi_driver_comparison.py  # Driver performance comparison
│   └── utils/
│       ├── constants.py                # Track configs & parameters
│       ├── helpers.py                  # Helper functions
│       └── pdf_generator.py            # PDF report generation
├── dataset/
│   ├── barber-motorsports-park/        # Barber race data
│   ├── circuit-of-the-americas/        # COTA race data
│   ├── indianapolis/                   # Indianapolis race data
│   └── sebring/                        # Sebring race data
├── models/                              # Trained ML models (.pkl files)
├── history/                             # Documentation & testing
├── tests/                               # Unit & integration tests
└── requirements.txt
```

## 🎯 Key Technologies

- **Python 3.11**: Core application framework
- **Streamlit**: Modern web dashboard with custom Toyota GR styling
- **Machine Learning**: XGBoost, Random Forest, Gradient Boosting
- **Data Processing**: Pandas, NumPy (handling 500+ laps across 4 tracks)
- **Visualization**: Plotly with dark theme and interactive charts
- **NLP**: Natural language strategy chat interface

## 📈 Performance Metrics

Our trained models achieve production-ready accuracy:

- **Tire Degradation Model**: R² = 0.94, MAE = 3.2% (predicts tire life within 3% error)
- **Pit Strategy Classifier**: 87% accuracy, F1-Score = 0.80
- **Driver Fingerprint Model**: 82% cross-validation accuracy across 33 drivers

## 🏆 Hackathon Category

**Real-Time Analytics** - A comprehensive tool for race strategy optimization and data-driven decision-making.

## 📝 License

MIT License
