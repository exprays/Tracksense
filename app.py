"""
Real-Time Race Strategy Optimizer - Main Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import RaceDataLoader
from src.data.preprocessor import RaceDataPreprocessor
from src.data.visualizer import RaceVisualizer
from src.models.tire_model import TireDegradationModel
from src.models.pit_optimizer import PitStopOptimizer
from src.models.fuel_model import FuelCalculator
from src.models.model_trainer import ModelTrainer
from src.models.ai_insights import RaceInsightsGenerator
from src.models.nlp_strategy import StrategyQueryEngine
from src.analytics.multi_driver_comparison import MultiDriverComparison
from src.utils.pdf_generator import generate_race_report
from src.utils.constants import TRACKS, DASHBOARD

# Page configuration
st.set_page_config(
    page_title="Toyota TrackSense",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0E1117;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #FAFAFA;
    }
    
    .big-font {
        font-size: 48px !important;
        font-weight: 800;
        background: linear-gradient(45deg, #EB0A1E, #FF5E6D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #3E404D;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        gap: 8px;
        height: 160px; /* Fixed height for consistency */
        justify-content: center;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: #EB0A1E;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: #A0A0A0;
        line-height: 1;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 600;
        color: #FAFAFA;
        line-height: 1;
        margin-top: 4px;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 14px;
        margin-top: 4px;
    }
    
    /* Custom Warning/Alert Boxes */
    .critical-warning {
        background-color: rgba(255, 68, 68, 0.1);
        border-left: 5px solid #ff4444;
        color: #ff4444;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
        display: flex;
        align-items: center;
    }
    
    .warning {
        background-color: rgba(255, 170, 0, 0.1);
        border-left: 5px solid #ffaa00;
        color: #ffaa00;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
        display: flex;
        align-items: center;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E;
        border-right: 1px solid #3E404D;
    }
    
    /* Tabs Styling - Menubar Style */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E1E1E;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #3E404D;
        gap: 4px;
        display: flex;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        height: 36px;
        background-color: transparent;
        border: none;
        border-radius: 4px;
        color: #A0A0A0;
        font-size: 14px;
        font-weight: 500;
        padding: 0 12px;
        flex: 0 1 auto;
        white-space: nowrap;
    }

    .stTabs [aria-selected="true"] {
        background-color: #EB0A1E;
        color: #FFFFFF;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF;
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    /* Plotly Chart Container */
    .js-plotly-plot .plotly .modebar {
        orientation: v;
        right: 0;
    }

    /* Select Box Styling */
    div[data-baseweb="select"] > div {
        background-color: transparent;
        border-color: #3E404D;
        border-radius: 6px;
        color: #FAFAFA;
        cursor: pointer;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #EB0A1E;
        cursor: pointer;
    }
    
    div[data-baseweb="popover"] {
        background-color: #1E1E1E;
        border: 1px solid #3E404D;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    div[data-baseweb="menu"] {
        background-color: #1E1E1E;
    }
    
    li[data-baseweb="menu-item"] {
        color: #A0A0A0;
        cursor: pointer;
    }
    
    li[data-baseweb="menu-item"]:hover, li[aria-selected="true"] {
        background-color: rgba(235, 10, 30, 0.1);
        color: #FAFAFA;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Initialize models"""
    tire_model = TireDegradationModel()
    pit_optimizer = PitStopOptimizer()
    fuel_calculator = FuelCalculator()
    visualizer = RaceVisualizer()
    insights_generator = RaceInsightsGenerator()
    
    # Try to load trained models
    models_path = Path(__file__).parent / 'models'
    if (models_path / 'tire_degradation_model.pkl').exists():
        try:
            tire_model.load(str(models_path / 'tire_degradation_model.pkl'))
        except Exception as e:
            st.warning(f"Could not load trained tire model: {e}")
    
    return tire_model, pit_optimizer, fuel_calculator, visualizer, insights_generator


@st.cache_resource
def load_data_loader():
    """Initialize data loader"""
    base_path = Path(__file__).parent / 'dataset'
    return RaceDataLoader(str(base_path))


def main():
    # Header
    col_header_1, col_header_2 = st.columns([3, 1])
    with col_header_1:
        st.markdown('<p class="big-font">Toyota Gazoo TrackSense</p>', unsafe_allow_html=True)
        st.markdown("**Toyota GR Cup Series - Data-Driven Strategy Tool**")
    with col_header_2:
        st.image("Toyota_Gazoo_Racing_emblem.svg", width=100)
    
    st.divider()
    
    # Initialize components
    loader = load_data_loader()
    preprocessor = RaceDataPreprocessor()
    tire_model, pit_optimizer, fuel_calculator, visualizer, insights_generator = load_models()
    
    # Sidebar - Race Selection
    with st.sidebar:
        st.header("🏎️ Race Configuration")
        
        with st.container():
            st.subheader("Event Details")
            track = st.selectbox(
                "Select Track",
                options=['barber', 'cota', 'indianapolis', 'sebring'],
                format_func=lambda x: TRACKS[x]['name'],
                key='track_selector'
            )
            
            race_number = st.selectbox(
                "Select Race",
                options=[1, 2],
                format_func=lambda x: f"Race {x}",
                key='race_selector'
            )
        
        st.divider()
        
        # Load available drivers
        available_drivers = loader.get_available_drivers(track, race_number)
        
        if not available_drivers:
            st.error(f"No driver data available for {TRACKS[track]['name']} Race {race_number}")
            st.info("This track/race combination may not have data files in the correct location. Please check the dataset folder.")
            return
        
        st.subheader("Driver Selection")
        selected_driver = st.selectbox(
            "Select Driver",
            options=available_drivers,
            format_func=lambda x: f"Car #{x}",
            key=f'driver_selector_{track}_{race_number}'
        )
    
    # Load and process data
    with st.spinner("Loading race data..."):
        driver_data = loader.get_driver_data(track, race_number, selected_driver)
        processed_data = preprocessor.process_driver_data(driver_data)
    
    if processed_data.empty:
        st.error("No data available for selected driver")
        return
    
    # Lap selection
    max_lap = int(processed_data['LAP_NUMBER'].max())
    min_lap = int(processed_data['LAP_NUMBER'].min())
    
    with st.sidebar:
        st.divider()
        st.subheader("⏱️ Simulation Control")
        # Only show slider if there's more than one lap
        if max_lap > min_lap:
            current_lap = st.slider(
                "Current Lap",
                min_value=min_lap,
                max_value=max_lap,
                value=max_lap,
                help="Simulate strategy at different points in the race"
            )
        else:
            current_lap = max_lap
            st.info(f"Only {max_lap} lap(s) of data available")
            
        st.caption(f"Simulating Lap {current_lap} of {max_lap}")
    
    # Filter data up to current lap
    current_data = processed_data[processed_data['LAP_NUMBER'] <= current_lap].copy()
    
    # Get current state
    current_state = preprocessor.get_current_state(current_data, current_lap)
    
    # Main dashboard layout
    with st.container():
        st.markdown("### 📊 Live Telemetry")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Lap", f"{current_lap}/{max_lap}", delta=None, delta_color="off")
        
        with col2:
            lap_time = current_state.get('last_lap_time', 0)
            prev_lap_time = current_state.get('prev_lap_time', 0) # Assuming this might exist or we can calculate
            delta = lap_time - prev_lap_time if prev_lap_time > 0 else 0
            st.metric("Last Lap Time", f"{lap_time:.3f}s", delta=f"{delta:+.3f}s" if delta != 0 else None, delta_color="inverse")
        
        with col3:
            tire_life = current_state.get('tire_life', 1.0) * 100
            tire_delta = f"{tire_life:.1f}%"
            # tire_color logic is handled by delta_color in st.metric usually, but we can use the value
            st.metric("Tire Life", tire_delta, delta="-1.2%" if current_lap > 1 else None, delta_color="normal") # Mock delta for now
        
        with col4:
            laps_of_fuel = current_state.get('laps_of_fuel', 0)
            st.metric("Fuel Remaining", f"{laps_of_fuel:.1f} laps", delta="-1.0 laps", delta_color="normal")
    
    st.divider()

    # Warnings section
    fuel_warnings = fuel_calculator.get_fuel_warnings(
        current_state,
        {'total_laps': max_lap}
    )
    
    if fuel_warnings or current_state.get('tire_warning', False):
        with st.expander("⚠️ Active Alerts", expanded=True):
            for warning in fuel_warnings:
                if warning['level'] == 'critical':
                    st.markdown(f'<div class="critical-warning">🛑 {warning["message"]} - {warning["action"]}</div>', 
                               unsafe_allow_html=True)
                elif warning['level'] == 'warning':
                    st.markdown(f'<div class="warning">⚠️ {warning["message"]} - {warning["action"]}</div>', 
                               unsafe_allow_html=True)
            
            if current_state.get('tire_warning', False):
                st.markdown('<div class="warning">⚠️ TIRE DEGRADATION WARNING - Consider pitting soon</div>', 
                           unsafe_allow_html=True)
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Overview", 
        "🔧 Pit Strategy", 
        "⏱️ Performance", 
        "⛽ Fuel Management",
        "🌡️ Conditions",
        "🤖 AI Insights",
        "🎓 Model Training",
        "🏆 Driver Comparison",
        "💬 Strategy Chat"
    ])
    
    with tab1:
        st.subheader("Race Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Lap time evolution
            fig = visualizer.plot_lap_times(current_data, f"Lap Times - Car #{selected_driver}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Dashboard summary
            fig = visualizer.create_dashboard_summary(current_state)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tire degradation
        fig = visualizer.plot_tire_degradation(current_data, f"Tire Analysis - Car #{selected_driver}")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Pit Stop Strategy")
        
        # Get pit recommendations
        pit_recommendation = pit_optimizer.get_real_time_recommendation(
            current_state,
            {'total_laps': max_lap}
        )
        
        pit_window = pit_recommendation['pit_window']
        
        # Pit window card
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Current Recommendation")
            
            if pit_window['should_pit']:
                urgency_color = {
                    'low': '🟢',
                    'high': '🟡',
                    'critical': '🔴'
                }
                st.markdown(f"**Urgency:** {urgency_color.get(pit_window['urgency'], '')} {pit_window['urgency'].upper()}")
                st.markdown(f"**Optimal Lap:** {pit_window['optimal_lap']}")
                st.markdown(f"**Window:** Laps {pit_window['window_start']}-{pit_window['window_end']}")
                st.markdown(f"**Reason:** {pit_window['reason']}")
            else:
                st.success("✅ No pit stop required")
                st.markdown(f"**Reason:** {pit_window['reason']}")
        
        with col2:
            # Pit recommendation visualization
            fig = visualizer.plot_pit_recommendation(current_data, "Pit Stop Decision")
            st.plotly_chart(fig, use_container_width=True)
        
        # Strategy comparison
        st.markdown("### Strategy Options")
        
        strategies = pit_recommendation.get('strategies', [])
        
        if strategies:
            for i, strategy in enumerate(strategies):
                with st.expander(f"{i+1}. {strategy['name']} - {strategy['description']}", 
                               expanded=(i==0)):
                    sim = strategy['simulation']
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Time", f"{sim['total_time']:.1f}s")
                    col2.metric("Pit Stops", sim['number_of_stops'])
                    col3.metric("Avg Stint", f"{sim['average_stint_length']:.1f} laps")
                    
                    if strategy['viable']:
                        st.success("✅ Viable strategy")
                    else:
                        st.error("❌ Not viable")
    
    with tab3:
        st.subheader("Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector comparison
            fig = visualizer.plot_sector_comparison(current_data, "Sector Time Evolution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Consistency analysis
            fig = visualizer.plot_consistency_analysis(current_data, "Driver Consistency")
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Consistency Score", f"{current_state.get('consistency', 100):.1f}/100")
        
        if 'DELTA_FROM_BEST' in current_data.columns and current_data['DELTA_FROM_BEST'].notna().any():
            avg_delta = current_data['DELTA_FROM_BEST'].mean()
            col2.metric("Avg Delta from Best", f"+{avg_delta:.3f}s")
        
        if 'TOP_SPEED' in current_data.columns:
            max_speed = current_data['TOP_SPEED'].max()
            col3.metric("Top Speed", f"{max_speed:.1f} km/h")
        
        deg_rate = current_state.get('degradation_rate', 0)
        col4.metric("Degradation Rate", f"+{deg_rate:.3f}s/lap")
    
    with tab4:
        st.subheader("Fuel Management")
        
        # Fuel visualization
        fig = visualizer.plot_fuel_consumption(current_data, "Fuel Status")
        st.plotly_chart(fig, use_container_width=True)
        
        # Fuel prediction
        st.markdown("### Fuel Strategy")
        
        fuel_pred = fuel_calculator.predict_fuel_to_finish(
            current_lap=current_lap,
            total_laps=max_lap,
            fuel_remaining=current_state.get('fuel_remaining', 0),
            fuel_per_lap=1.2
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if fuel_pred['can_finish']:
                st.success(f"✅ Can finish race with {fuel_pred['margin_laps']:.1f} laps margin")
                st.markdown(f"**Confidence:** {fuel_pred['confidence'].upper()}")
            else:
                st.error("❌ Insufficient fuel to finish")
                st.markdown(f"**Shortfall:** {abs(fuel_pred['margin_laps']):.1f} laps")
        
        with col2:
            # Fuel save mode
            fuel_save = fuel_calculator.calculate_fuel_save_mode(
                current_fuel=current_state.get('fuel_remaining', 0),
                laps_remaining=max_lap - current_lap
            )
            
            if fuel_save['fuel_save_required']:
                st.warning(f"⚠️ {fuel_save['message']}")
                st.markdown(f"**Required Saving:** {fuel_save['saving_percentage']:.1f}%")
            else:
                st.success("✅ No fuel saving required")
    
    with tab5:
        st.subheader("Weather & Track Conditions")
        
        if 'AIR_TEMP' in current_data.columns:
            fig = visualizer.plot_weather_impact(current_data, "Weather Conditions")
            st.plotly_chart(fig, use_container_width=True)
            
            # Current conditions
            st.markdown("### Current Conditions")
            col1, col2, col3, col4 = st.columns(4)
            
            if 'AIR_TEMP' in current_data.columns:
                air_temp = current_data.iloc[-1]['AIR_TEMP']
                col1.metric("Air Temp", f"{air_temp:.1f}°C")
            
            if 'HUMIDITY' in current_data.columns:
                humidity = current_data.iloc[-1]['HUMIDITY']
                col2.metric("Humidity", f"{humidity:.1f}%")
            
            if 'WIND_SPEED' in current_data.columns:
                wind = current_data.iloc[-1]['WIND_SPEED']
                col3.metric("Wind Speed", f"{wind:.1f} km/h")
            
            if 'RAIN' in current_data.columns:
                rain = current_data.iloc[-1]['RAIN']
                rain_status = "Yes" if rain > 0 else "No"
                col4.metric("Rain", rain_status)
        else:
            st.info("Weather data not available for this session")
    
    with tab6:
        st.subheader("🤖 AI-Powered Race Insights")
        
        # Get tire predictions
        tire_predictions = tire_model.predict_next_laps(current_data, n_laps=5)
        
        # Generate comprehensive insights
        insights = insights_generator.generate_comprehensive_insights(
            current_state=current_state,
            race_context={'total_laps': max_lap, 'position': None},
            processed_data=current_data,
            tire_predictions=tire_predictions,
            pit_window=pit_recommendation['pit_window'],
            strategies=pit_recommendation.get('strategies', [])
        )
        
        # Display priority recommendations
        if insights.get('priority_recommendations'):
            st.markdown("### 🎯 Priority Actions")
            urgency_colors = {
                'critical': '#ff4444',
                'high': '#ffaa00',
                'low': '#44ff44'
            }
            urgency = insights.get('overall_urgency', 'low')
            st.markdown(
                f'<div style="background-color: {urgency_colors[urgency]}; '
                f'padding: 15px; border-radius: 5px; margin: 10px 0;">'
                f'<b>Overall Urgency: {urgency.upper()}</b></div>',
                unsafe_allow_html=True
            )
            
            for i, rec in enumerate(insights['priority_recommendations'][:5], 1):
                st.markdown(f"**{i}.** {rec}")
        
        st.markdown("---")
        
        # Display insights by category
        for section in insights.get('sections', []):
            with st.expander(f"{section['category']} (Confidence: {section.get('confidence', 0)*100:.0f}%)", expanded=True):
                st.markdown("**Analysis:**")
                for insight in section.get('insights', []):
                    st.markdown(f"• {insight}")
                
                if section.get('recommendations'):
                    st.markdown("\n**Recommendations:**")
                    for rec in section['recommendations']:
                        st.markdown(f"→ {rec}")
                
                # Display metrics if available
                if section.get('metrics'):
                    st.markdown("\n**Metrics:**")
                    cols = st.columns(len(section['metrics']))
                    for col, (key, value) in zip(cols, section['metrics'].items()):
                        if isinstance(value, float):
                            col.metric(key.replace('_', ' ').title(), f"{value:.3f}")
                        else:
                            col.metric(key.replace('_', ' ').title(), value)
                
                # Display predictions if available
                if section.get('predictions'):
                    st.markdown("\n**Predictions:**")
                    for key, value in section['predictions'].items():
                        if isinstance(value, (int, float)):
                            st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
    
    with tab7:
        st.subheader("🎓 Model Training & Performance")
        
        st.markdown("""
        This section allows you to train machine learning models on all available race data 
        to improve prediction accuracy for tire degradation, pit strategy, and driver performance.
        """)
        
        models_path = Path(__file__).parent / 'models'
        history_path = models_path / 'training_history.json'
        
        # Load training history if available
        training_history = None
        if history_path.exists():
            import json
            with open(history_path, 'r') as f:
                training_history = json.load(f)
        
        # Check for existing trained models with detailed info
        st.markdown("### 📊 Current Model Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tire_model_exists = (models_path / 'tire_degradation_model.pkl').exists()
            st.markdown("**🔵 Tire Degradation Model**")
            if tire_model_exists:
                st.success("✓ Trained")
                st.caption("**Type:** Regression Model")
                st.caption("**Algorithm:** Gradient Boosting")
                if training_history and training_history.get('tire_model'):
                    tire_metrics = training_history['tire_model']['metrics']
                    st.metric("R² Score", f"{tire_metrics['r2_score']:.3f}", 
                             help="1.0 = perfect predictions")
                    st.metric("MAE", f"{tire_metrics['mae']:.4f}",
                             help="Average prediction error")
            else:
                st.warning("⚠️ Not Trained")
                st.caption("**Type:** Regression Model")
                st.caption("**Algorithm:** Gradient Boosting")
                st.info("Train to enable tire life predictions")
        
        with col2:
            pit_model_exists = (models_path / 'pit_strategy_model.pkl').exists()
            st.markdown("**🟢 Pit Strategy Model**")
            if pit_model_exists:
                st.success("✓ Trained")
                st.caption("**Type:** Classification Model")
                st.caption("**Algorithm:** XGBoost/RandomForest")
                if training_history and training_history.get('pit_strategy_model'):
                    pit_metrics = training_history['pit_strategy_model']
                    st.metric("Accuracy", f"{pit_metrics['val_accuracy']:.3f}",
                             help="Validation set accuracy")
                    st.metric("F1 Score", f"{pit_metrics['f1_score']:.3f}",
                             help="Precision-recall balance")
            else:
                st.warning("⚠️ Not Trained")
                st.caption("**Type:** Classification Model")
                st.caption("**Algorithm:** XGBoost/RandomForest")
                st.info("Train to enable pit timing predictions")
        
        with col3:
            driver_model_exists = (models_path / 'driver_fingerprint_model.pkl').exists()
            st.markdown("**🟡 Driver Fingerprint Model**")
            if driver_model_exists:
                st.success("✓ Trained")
                st.caption("**Type:** Multi-class Classification")
                st.caption("**Algorithm:** Random Forest")
                if training_history and training_history.get('driver_model'):
                    driver_metrics = training_history['driver_model']
                    st.metric("CV Accuracy", f"{driver_metrics['cv_accuracy_mean']:.3f}",
                             help="Cross-validation accuracy")
                    st.metric("Drivers", driver_metrics['n_drivers'],
                             help="Unique drivers analyzed")
            else:
                st.warning("⚠️ Not Trained")
                st.caption("**Type:** Multi-class Classification")
                st.caption("**Algorithm:** Random Forest")
                st.info("Train to enable driver analysis")
        
        st.markdown("---")
        
        # Training section
        st.markdown("### Train Models")
        
        st.warning("""
        ⚠️ **Warning**: Training will use all available race data from both tracks. 
        This process may take 1-2 minutes to complete.
        """)
        
        if st.button("🚀 Train All Models", type="primary", use_container_width=True):
            with st.spinner("Training models... This may take a minute."):
                try:
                    # Initialize trainer
                    trainer = ModelTrainer(
                        data_path=str(Path(__file__).parent / 'dataset'),
                        models_output_path=str(models_path)
                    )
                    
                    # Train all models
                    results = trainer.train_all_models()
                    
                    st.success("✓ All models trained successfully!")
                    
                    # Display results
                    st.markdown("### Training Results")
                    
                    # Tire Model Results
                    if results.get('tire_model'):
                        with st.expander("📊 Tire Degradation Model", expanded=True):
                            tire = results['tire_model']
                            
                            st.markdown("**Model Type:** Gradient Boosting Regressor")
                            st.markdown("**Parameters:** n_estimators=100, learning_rate=0.1, max_depth=4")
                            st.markdown("**Task:** Predict tire life (0-1 scale) based on lap data, weather, and driving style")
                            st.markdown("")
                            
                            st.markdown("**Performance Metrics:**")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric("R² Score", f"{tire['metrics']['r2_score']:.4f}", 
                                       help="Coefficient of determination (1.0 = perfect predictions)")
                            col2.metric("MAE", f"{tire['metrics']['mae']:.4f}",
                                       help="Mean Absolute Error - average prediction error")
                            col3.metric("RMSE", f"{tire['metrics']['rmse']:.4f}",
                                       help="Root Mean Square Error - penalizes large errors")
                            col4.metric("Total Laps", tire['total_laps'],
                                       help="Number of training samples")
                            
                            if tire.get('feature_importance'):
                                st.markdown("")
                                st.markdown("**Top 5 Important Features:**")
                                top_features = list(tire['feature_importance'].items())[:5]
                                for i, (feat, importance) in enumerate(top_features, 1):
                                    st.markdown(f"{i}. **{feat}**: {importance:.4f}")
                    
                    # Pit Strategy Model Results
                    if results.get('pit_strategy_model'):
                        with st.expander("🔧 Pit Strategy Model", expanded=True):
                            pit = results['pit_strategy_model']
                            
                            st.markdown("**Model Type:** XGBoost Classifier (or RandomForest fallback)")
                            st.markdown("**Parameters:** n_estimators=100, max_depth=5, learning_rate=0.1")
                            st.markdown("**Task:** Binary classification - predict whether to pit (Yes/No)")
                            st.markdown("**Input Features:** Tire life, fuel remaining, degradation rate, consistency, track, weather")
                            st.markdown("")
                            
                            st.markdown("**Classification Metrics:**")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            col1.metric("Train Accuracy", f"{pit['train_accuracy']:.4f}",
                                       help="Accuracy on training set")
                            col2.metric("Val Accuracy", f"{pit['val_accuracy']:.4f}",
                                       help="Accuracy on validation set (unseen data)")
                            col3.metric("Precision", f"{pit['precision']:.4f}",
                                       help="When model says 'pit', how often is it correct?")
                            col4.metric("Recall", f"{pit['recall']:.4f}",
                                       help="Of all times you should pit, how many does model catch?")
                            col5.metric("F1 Score", f"{pit['f1_score']:.4f}",
                                       help="Harmonic mean of precision and recall")
                            
                            st.markdown("")
                            st.markdown(f"**Training Samples:** {pit['n_samples']:,} lap decisions")
                    
                    # Driver Model Results
                    if results.get('driver_model'):
                        with st.expander("👤 Driver Fingerprint Model", expanded=True):
                            driver = results['driver_model']
                            
                            st.markdown("**Model Type:** Random Forest Classifier")
                            st.markdown("**Parameters:** n_estimators=100, max_depth=10")
                            st.markdown("**Task:** Multi-class classification - identify driver by their driving style")
                            st.markdown("**Features:** Lap time, consistency, aggression, sector strengths, speed characteristics")
                            st.markdown("")
                            
                            st.markdown("**Classification Metrics:**")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric("CV Accuracy", 
                                      f"{driver['cv_accuracy_mean']:.4f}",
                                      help="Cross-validation accuracy (mean across folds)")
                            col2.metric("Std Dev", 
                                      f"± {driver['cv_accuracy_std']:.4f}",
                                      help="Standard deviation of CV scores")
                            col3.metric("Drivers", driver['n_drivers'],
                                       help="Number of unique drivers analyzed")
                            col4.metric("Samples", driver['n_samples'],
                                       help="Number of race sessions used")
                            
                            st.markdown("")
                            st.markdown(f"**Drivers Analyzed:** {', '.join(map(str, driver['drivers'][:10]))}")
                            
                            if driver.get('feature_importance'):
                                st.markdown("")
                                st.markdown("**Key Driving Style Features:**")
                                top_features = sorted(driver['feature_importance'].items(), 
                                                    key=lambda x: x[1], reverse=True)[:5]
                                for i, (feat, importance) in enumerate(top_features, 1):
                                    st.markdown(f"{i}. **{feat.replace('_', ' ').title()}**: {importance:.4f}")
                    
                    # Generate and display report
                    st.markdown("### 📄 Training Report")
                    report = trainer.generate_training_report()
                    st.code(report, language=None)
                    
                    st.info("💡 Reload the page to use the newly trained models in predictions.")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.exception(e)
        
        # Display training history with visualizations if available
        if training_history:
            st.markdown("---")
            st.markdown("### 📜 Training History & Performance Visualization")
            
            st.markdown(f"**Last Training:** {training_history.get('timestamp', 'Unknown')}")
            st.markdown(f"**Races Used:** {len(training_history.get('races_used', []))}")
            
            # Create performance comparison visualization
            import plotly.graph_objects as go
            
            # Prepare data for visualization
            models_data = []
            colors_map = {'Tire Model': '#636EFA', 'Pit Strategy': '#00CC96', 'Driver Model': '#FFA15A'}
            
            if training_history.get('tire_model'):
                tire = training_history['tire_model']
                models_data.append({
                    'name': 'Tire Model',
                    'type': 'Regression',
                    'r2': tire['metrics']['r2_score'],
                    'mae': tire['metrics']['mae'],
                    'samples': tire['total_laps']
                })
            
            if training_history.get('pit_strategy_model'):
                pit = training_history['pit_strategy_model']
                models_data.append({
                    'name': 'Pit Strategy',
                    'type': 'Classification',
                    'accuracy': pit['val_accuracy'],
                    'f1': pit['f1_score'],
                    'samples': pit['n_samples']
                })
            
            if training_history.get('driver_model'):
                driver = training_history['driver_model']
                models_data.append({
                    'name': 'Driver Model',
                    'type': 'Multi-class',
                    'accuracy': driver['cv_accuracy_mean'],
                    'samples': driver['n_samples']
                })
            
            if models_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Model Performance Comparison
                    fig1 = go.Figure()
                    
                    for model in models_data:
                        if 'r2' in model:
                            fig1.add_trace(go.Bar(
                                name=model['name'],
                                x=['R² Score'],
                                y=[model['r2']],
                                marker_color=colors_map[model['name']],
                                text=[f"{model['r2']:.3f}"],
                                textposition='auto',
                            ))
                        elif 'accuracy' in model:
                            fig1.add_trace(go.Bar(
                                name=model['name'],
                                x=['Accuracy'],
                                y=[model['accuracy']],
                                marker_color=colors_map[model['name']],
                                text=[f"{model['accuracy']:.3f}"],
                                textposition='auto',
                            ))
                    
                    fig1.update_layout(
                        title="Model Performance Scores",
                        yaxis_title="Score",
                        yaxis=dict(range=[0, 1.1]),
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Training Sample Distribution
                    fig2 = go.Figure(data=[
                        go.Bar(
                            x=[m['name'] for m in models_data],
                            y=[m['samples'] for m in models_data],
                            marker_color=[colors_map[m['name']] for m in models_data],
                            text=[f"{m['samples']:,}" for m in models_data],
                            textposition='auto',
                        )
                    ])
                    
                    fig2.update_layout(
                        title="Training Samples per Model",
                        yaxis_title="Number of Samples",
                        height=350
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Model Type Summary
                st.markdown("### 🎯 Model Type Summary")
                summary_cols = st.columns(3)
                
                with summary_cols[0]:
                    st.info("**Regression Model**\n\n📊 Tire Degradation\n\n• Predicts continuous values (0-1)\n• Gradient Boosting algorithm")
                
                with summary_cols[1]:
                    st.success("**Binary Classification**\n\n🔧 Pit Strategy\n\n• Predicts Yes/No decisions\n• XGBoost/RandomForest")
                
                with summary_cols[2]:
                    st.warning("**Multi-class Classification**\n\n👤 Driver Fingerprint\n\n• Identifies 33+ drivers\n• Random Forest algorithm")
            
            if st.checkbox("Show detailed training history JSON"):
                st.json(training_history)
    
    # Tab 8: Multi-Driver Comparison
    with tab8:
        st.subheader("🏆 Multi-Driver Performance Comparison")
        st.write("Compare multiple drivers' performance across key metrics")
        
        # Load all drivers' data for comparison
        if loader and preprocessor:
            st.markdown("### Select Drivers to Compare")
            
            # Multi-select for drivers
            comparison_drivers = st.multiselect(
                "Choose 2 or more drivers",
                options=available_drivers,
                default=available_drivers[:min(3, len(available_drivers))],
                help="Select drivers to compare side-by-side",
                format_func=lambda x: f"Car #{x}"
            )
            
            if len(comparison_drivers) >= 2:
                # Load data for all selected drivers
                with st.spinner("Loading comparison data..."):
                    all_driver_data = []
                    for driver_no in comparison_drivers:
                        driver_data = loader.get_driver_data(track, race_number, driver_no)
                        driver_processed = preprocessor.process_driver_data(driver_data)
                        if not driver_processed.empty:
                            driver_processed['NO'] = driver_no  # Add driver number column
                            all_driver_data.append(driver_processed)
                    
                    if all_driver_data:
                        combined_data = pd.concat(all_driver_data, ignore_index=True)
                        multi_comp = MultiDriverComparison(combined_data)
                        
                        selected_drivers = comparison_drivers
                        
                        # Performance comparison
                        st.markdown("### 📊 Overall Performance")
                        perf_df = multi_comp.get_performance_comparison(selected_drivers)
                        
                        if not perf_df.empty:
                            # Check which columns exist
                            highlight_cols = []
                            if 'Best Lap' in perf_df.columns:
                                highlight_cols.append('Best Lap')
                            if 'Avg Lap' in perf_df.columns:
                                highlight_cols.append('Avg Lap')
                            
                            if highlight_cols:
                                st.dataframe(perf_df.style.highlight_min(
                                    subset=highlight_cols,
                                    color='lightgreen'
                                ), use_container_width=True)
                            else:
                                st.dataframe(perf_df, use_container_width=True)
                            
                            # Lap time evolution chart
                            st.markdown("### 📈 Lap Time Evolution")
                            fig = multi_comp.plot_lap_time_evolution(selected_drivers)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Consistency comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🎯 Consistency Comparison")
                            consist_df = multi_comp.get_consistency_comparison(selected_drivers)
                            if not consist_df.empty:
                                st.dataframe(consist_df.style.highlight_max(
                                    subset=['Consistency Score'],
                                    color='lightgreen'
                                ), use_container_width=True)
                        
                        with col2:
                            st.markdown("### 🏁 Sector Performance")
                            sector_df = multi_comp.get_sector_comparison(selected_drivers)
                            if not sector_df.empty:
                                st.dataframe(sector_df, use_container_width=True)
                        
                        # Tire degradation comparison
                        st.markdown("### 🛞 Tire Degradation Comparison")
                        fig = multi_comp.plot_tire_degradation_comparison(selected_drivers)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Head-to-head comparison
                        if len(selected_drivers) == 2:
                            st.markdown("### ⚔️ Head-to-Head Analysis")
                            max_lap_comparison = int(combined_data['LAP_NUMBER'].max())
                            lap_number = st.slider(
                                "Select lap to compare",
                                min_value=1,
                                max_value=max_lap_comparison,
                                value=1
                            )
                            
                            h2h_result = multi_comp.get_head_to_head(
                                selected_drivers[0],
                                selected_drivers[1],
                                lap_number
                            )
                            
                            if h2h_result['valid']:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        f"Car #{selected_drivers[0]}",
                                        f"{h2h_result['driver1_time']:.3f}s"
                                    )
                                
                                with col2:
                                    diff = h2h_result['time_difference']
                                    st.metric(
                                        "Time Difference",
                                        f"{abs(diff):.3f}s",
                                        delta=f"Car #{h2h_result['faster_driver']} faster"
                                    )
                                
                                with col3:
                                    st.metric(
                                        f"Car #{selected_drivers[1]}",
                                        f"{h2h_result['driver2_time']:.3f}s"
                                    )
                            else:
                                st.info("No lap time data available for selected lap.")
                    else:
                        st.warning("No data available for selected drivers.")
                
            elif len(comparison_drivers) == 1:
                st.info("Please select at least 2 drivers to compare.")
            else:
                st.info("Please select drivers from the list above.")
        else:
            st.warning("Please load race data from the sidebar first.")
    
    # Tab 9: Natural Language Strategy Chat
    with tab9:
        st.subheader("💬 Strategy Chat Assistant")
        st.write("Ask questions about race strategy in plain English!")
        
        # Initialize NLP engine
        if 'nlp_engine' not in st.session_state:
            st.session_state.nlp_engine = StrategyQueryEngine()
            st.session_state.chat_history = []
        
        nlp_engine = st.session_state.nlp_engine
        
        # Set context if data is available
        if preprocessor and processed_data is not None and current_state:
            race_context = {
                'total_laps': 30,  # Default, can be made configurable
                'track': track,
                'race': race_number
            }
            nlp_engine.set_context(current_state, processed_data, race_context)
        
        # Suggested questions
        st.markdown("### 💡 Suggested Questions")
        suggestions = nlp_engine.get_suggested_questions()
        
        cols = st.columns(4)
        for idx, suggestion in enumerate(suggestions):
            with cols[idx % 4]:
                if st.button(suggestion, key=f"suggest_{idx}"):
                    st.session_state.chat_history.append({
                        'query': suggestion,
                        'answer': nlp_engine.process_query(suggestion)
                    })
        
        st.markdown("---")
        
        # Chat interface
        st.markdown("### 🗨️ Ask Your Question")
        
        user_query = st.text_input(
            "Type your question:",
            placeholder="e.g., How are my tires doing?",
            key="user_query"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🚀 Ask", type="primary")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and user_query:
            if not nlp_engine.context:
                st.warning("⚠️ Please load race data from the sidebar first to enable strategy chat.")
            else:
                answer = nlp_engine.process_query(user_query)
                st.session_state.chat_history.append({
                    'query': user_query,
                    'answer': answer
                })
        
        # Display chat history (most recent first)
        if st.session_state.chat_history:
            st.markdown("### 📜 Conversation History")
            
            for idx, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"**Q{len(st.session_state.chat_history) - idx}:** {chat['query']}")
                    st.markdown(chat['answer'])
                    st.markdown("---")
        else:
            st.info("👋 No questions asked yet. Try one of the suggested questions above or type your own!")
    
    # Add PDF Export button to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Export Options")
    
    if st.sidebar.button("📥 Generate PDF Report", type="primary"):
        if processed_data is not None and current_state:
            try:
                # Generate insights for the report
                insights_gen = RaceInsightsGenerator()
                
                # Prepare tire predictions
                tire_predictions = {
                    'future_predictions': [],
                    'pit_lap_recommendation': current_state.get('lap', 0) + 5
                }
                
                # Prepare race context
                race_ctx = {
                    'total_laps': max_lap,
                    'current_position': 1
                }
                
                insights = insights_gen.generate_comprehensive_insights(
                    current_state=current_state,
                    race_context=race_ctx,
                    processed_data=processed_data,
                    tire_predictions=tire_predictions,
                    pit_window={'optimal_start': 8, 'optimal_end': 12},
                    strategies=[]
                )
                
                # Create comparison data if multiple drivers
                comparison_data = None
                if 'NO' in processed_data.columns:
                    all_drivers = processed_data['NO'].unique()
                    if len(all_drivers) > 1:
                        multi_comp = MultiDriverComparison(processed_data)
                        comparison_data = multi_comp.get_performance_comparison(
                            all_drivers[:min(5, len(all_drivers))]
                        )
                
                # Generate PDF
                pdf_bytes = generate_race_report(
                    driver_number=selected_driver,
                    track_name=TRACKS.get(track, {}).get('name', track),
                    race_number=race_number,
                    current_state=current_state,
                    processed_data=processed_data,
                    insights=insights,
                    comparison_data=comparison_data
                )
                
                # Download button
                st.sidebar.download_button(
                    label="💾 Download Report",
                    data=pdf_bytes,
                    file_name=f"race_strategy_report_car{selected_driver}_{track}_R{race_number}.pdf",
                    mime="application/pdf"
                )
                
                st.sidebar.success("✓ PDF report generated successfully!")
                
            except Exception as e:
                st.sidebar.error(f"Error generating PDF: {str(e)}")
        else:
            st.sidebar.warning("Please load race data first to generate a report.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Real-Time Race Strategy Optimizer** | Toyota GR Cup Series | Hack the Track 2025")


if __name__ == "__main__":
    main()
