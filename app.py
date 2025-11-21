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
from src.utils.constants import TRACKS, DASHBOARD

# Page configuration
st.set_page_config(
    page_title="Race Strategy Optimizer",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .critical-warning {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    .warning {
        background-color: #ffaa00;
        color: black;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
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
    
    return tire_model, pit_optimizer, fuel_calculator, visualizer


@st.cache_resource
def load_data_loader():
    """Initialize data loader"""
    base_path = Path(__file__).parent / 'dataset'
    return RaceDataLoader(str(base_path))


def main():
    # Header
    st.markdown('<p class="big-font">🏁 Real-Time Race Strategy Optimizer</p>', unsafe_allow_html=True)
    st.markdown("**Toyota GR Cup Series - Data-Driven Strategy Tool**")
    
    # Initialize components
    loader = load_data_loader()
    preprocessor = RaceDataPreprocessor()
    tire_model, pit_optimizer, fuel_calculator, visualizer = load_models()
    
    # Sidebar - Race Selection
    st.sidebar.title("Race Selection")
    
    track = st.sidebar.selectbox(
        "Select Track",
        options=['barber', 'cota'],
        format_func=lambda x: TRACKS[x]['name']
    )
    
    race_number = st.sidebar.selectbox(
        "Select Race",
        options=[1, 2],
        format_func=lambda x: f"Race {x}"
    )
    
    # Load available drivers
    available_drivers = loader.get_available_drivers(track, race_number)
    
    if not available_drivers:
        st.error("No driver data available for selected race")
        return
    
    selected_driver = st.sidebar.selectbox(
        "Select Driver",
        options=available_drivers,
        format_func=lambda x: f"Car #{x}"
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
    current_lap = st.sidebar.slider(
        "Current Lap",
        min_value=1,
        max_value=max_lap,
        value=max_lap,
        help="Simulate strategy at different points in the race"
    )
    
    # Filter data up to current lap
    current_data = processed_data[processed_data['LAP_NUMBER'] <= current_lap].copy()
    
    # Get current state
    current_state = preprocessor.get_current_state(current_data, current_lap)
    
    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Lap", f"{current_lap}/{max_lap}")
    
    with col2:
        lap_time = current_state.get('last_lap_time', 0)
        st.metric("Last Lap Time", f"{lap_time:.3f}s" if lap_time > 0 else "N/A")
    
    with col3:
        tire_life = current_state.get('tire_life', 1.0) * 100
        tire_delta = f"{tire_life:.1f}%"
        tire_color = "🟢" if tire_life > 75 else "🟡" if tire_life > 65 else "🔴"
        st.metric(f"{tire_color} Tire Life", tire_delta)
    
    with col4:
        laps_of_fuel = current_state.get('laps_of_fuel', 0)
        fuel_color = "🟢" if laps_of_fuel > 5 else "🟡" if laps_of_fuel > 3 else "🔴"
        st.metric(f"{fuel_color} Fuel Remaining", f"{laps_of_fuel:.1f} laps")
    
    # Warnings section
    fuel_warnings = fuel_calculator.get_fuel_warnings(
        current_state,
        {'total_laps': max_lap}
    )
    
    for warning in fuel_warnings:
        if warning['level'] == 'critical':
            st.markdown(f'<div class="critical-warning">⚠️ {warning["message"]} - {warning["action"]}</div>', 
                       unsafe_allow_html=True)
        elif warning['level'] == 'warning':
            st.markdown(f'<div class="warning">⚠️ {warning["message"]} - {warning["action"]}</div>', 
                       unsafe_allow_html=True)
    
    if current_state.get('tire_warning', False):
        st.markdown('<div class="warning">⚠️ TIRE DEGRADATION WARNING - Consider pitting soon</div>', 
                   unsafe_allow_html=True)
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔧 Pit Strategy", 
        "⏱️ Performance", 
        "⛽ Fuel Management",
        "🌡️ Conditions"
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
    
    # Footer
    st.markdown("---")
    st.markdown("**Real-Time Race Strategy Optimizer** | Toyota GR Cup Series | Hack the Track 2025")


if __name__ == "__main__":
    main()
