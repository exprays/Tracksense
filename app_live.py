"""
Live Race Analytics Dashboard
Real-time telemetry visualization and ML predictions

NOTE: This dashboard reads from the in-memory buffer.
You must run the receiver in a separate terminal:
    python -m src.realtime.receiver
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from src.realtime.file_buffer import get_file_buffer


# Page config
st.set_page_config(
    page_title="Toyota GR Cup - Live Race Analytics",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Toyota GR Theme CSS (same as app.py)
st.markdown("""
<style>
    /* Main theme */
    :root {
        --toyota-red: #EB0A1E;
        --dark-bg: #0E1117;
        --card-bg: #1E1E1E;
    }
    
    /* Live indicator */
    .live-badge {
        display: inline-block;
        background: var(--toyota-red);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: var(--card-bg);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    [data-testid="metric-container"]:hover {
        border-color: var(--toyota-red);
        box-shadow: 0 4px 12px rgba(235,10,30,0.3);
    }
    
    /* Strategy alerts */
    .alert-box {
        background: var(--card-bg);
        border-left: 4px solid var(--toyota-red);
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <h1 style='text-align: center;'>
        🏁 Toyota GR Cup - Live Race Analytics 
        <span class='live-badge'>● LIVE</span>
    </h1>
""", unsafe_allow_html=True)

# Get file-based buffer instance (shared across processes)
buffer = get_file_buffer()

# Sidebar - Session Selection
st.sidebar.title("📡 Live Sessions")

# Check for active sessions to infer receiver status
sessions = buffer.get_all_sessions()

if sessions:
    st.sidebar.success("🟢 Receiver: Active")
else:
    st.sidebar.warning("⚠️ Receiver: No sessions\n\nStart receiver:\n```bash\npython -m src.realtime.receiver\n```")

if not sessions:
    st.sidebar.info("⏳ Waiting for simulator connection...")
    st.info("### No active sessions\n\nStart the Go simulator to begin streaming telemetry:\n\n```bash\ncd simulator\ngo run . -mode csv -csv ../dataset/barber-motorsports-park/barber/R1_barber_telemetry_data.csv\n```")
    st.stop()

selected_session = st.sidebar.selectbox("Select Session", sessions)

# Get cars in session
cars = buffer.get_all_cars(selected_session)

if not cars:
    st.warning("No cars in this session yet...")
    st.stop()

selected_car = st.sidebar.selectbox("Select Car", cars)

# Auto-refresh settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh rate (Hz)", 1, 10, 5)

# Session stats
stats = buffer.get_session_stats(selected_session)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Session Stats")
st.sidebar.metric("Active Cars", stats["num_cars"])

# Main dashboard
latest = buffer.get_latest(selected_session, selected_car)
prediction = buffer.get_prediction(selected_session, selected_car)

if not latest:
    st.warning("Waiting for telemetry data...")
    st.stop()

# Extract data
position = latest.get("position", {})
dynamics = latest.get("dynamics", {})

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Lap", 
        position.get("lap", "-"),
        delta=None
    )

with col2:
    lap_time = prediction.get("predicted_lap_time") if prediction else None
    if lap_time:
        st.metric(
            "Predicted Lap Time",
            f"{lap_time:.2f}s",
            delta=f"±{lap_time * 0.05:.1f}s",
            delta_color="off"
        )
    else:
        st.metric("Predicted Lap Time", "Calculating...")

with col3:
    tire_deg = prediction.get("tire_deg_pct") if prediction else 0
    st.metric(
        "Tire Degradation",
        f"{tire_deg:.1f}%",
        delta=f"+{tire_deg * 0.02:.1f}% per lap" if tire_deg > 0 else None,
        delta_color="inverse"
    )

with col4:
    overtake_prob = prediction.get("overtake_prob_next_3_laps") if prediction else 0
    st.metric(
        "Overtake Probability",
        f"{overtake_prob:.1%}",
        delta="Next 3 laps",
        delta_color="normal" if overtake_prob > 0.5 else "off"
    )

st.markdown("---")

# Live telemetry gauges
st.subheader("🎛️ Live Telemetry")

col1, col2 = st.columns(2)

with col1:
    # Speed gauge
    fig_speed = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dynamics.get("speed_kmh", 0),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Speed (km/h)", 'font': {'color': 'white'}},
        delta={'reference': 150, 'increasing': {'color': "#00FF00"}},
        gauge={
            'axis': {'range': [None, 250], 'tickcolor': "white"},
            'bar': {'color': "#EB0A1E"},
            'bgcolor': "#1E1E1E",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 100], 'color': '#333'},
                {'range': [100, 200], 'color': '#444'},
                {'range': [200, 250], 'color': '#555'}
            ],
            'threshold': {
                'line': {'color': "#00FF00", 'width': 4},
                'thickness': 0.75,
                'value': 200
            }
        }
    ))
    fig_speed.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={'color': "white", 'family': "Arial"},
        height=300
    )
    st.plotly_chart(fig_speed, use_container_width=True)

with col2:
    # RPM gauge
    fig_rpm = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dynamics.get("rpm", 0),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "RPM", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [None, 8000], 'tickcolor': "white"},
            'bar': {'color': "#EB0A1E"},
            'bgcolor': "#1E1E1E",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 4000], 'color': '#333'},
                {'range': [4000, 6500], 'color': '#444'},
                {'range': [6500, 8000], 'color': '#FF4444'}
            ],
            'threshold': {
                'line': {'color': "#FFA500", 'width': 4},
                'thickness': 0.75,
                'value': 7000
            }
        }
    ))
    fig_rpm.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={'color': "white", 'family': "Arial"},
        height=300
    )
    st.plotly_chart(fig_rpm, use_container_width=True)

# Control inputs
col1, col2, col3 = st.columns(3)

with col1:
    throttle = dynamics.get("throttle_pedal_pct", 0)
    st.metric("Throttle", f"{throttle:.1f}%")
    st.progress(throttle / 100.0)

with col2:
    brake = dynamics.get("brake_pressure_front_bar", 0)
    st.metric("Brake Pressure", f"{brake:.1f} bar")
    st.progress(min(brake / 50.0, 1.0))

with col3:
    st.metric("Gear", dynamics.get("gear", "-"))
    st.metric("Steering", f"{dynamics.get('steering_angle_deg', 0):.1f}°")

st.markdown("---")

# Strategy alerts
if prediction and prediction.get("strategy_alerts"):
    st.subheader("🎯 Strategy Alerts")
    
    for alert in prediction["strategy_alerts"]:
        st.markdown(f"""
            <div class='alert-box'>{alert}</div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ML Predictions panel
st.subheader("🤖 AI Predictions")

if prediction:
    col1, col2 = st.columns(2)
    
    with col1:
        incident_risk = prediction.get("incident_risk", 0)
        risk_level = prediction.get("incident_risk_level", "low")
        
        risk_color = {
            "low": "#00FF00",
            "medium": "#FFA500",
            "high": "#FF0000"
        }.get(risk_level, "#888")
        
        st.markdown(f"""
            <div style='background: {risk_color}22; border: 2px solid {risk_color}; 
                        border-radius: 8px; padding: 16px; margin: 8px 0;'>
                <h4 style='margin: 0; color: {risk_color};'>Incident Risk: {risk_level.upper()}</h4>
                <p style='margin: 8px 0 0 0; font-size: 24px;'>{incident_risk:.1%}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence = prediction.get("lap_time_confidence", 0.85)
        st.markdown(f"""
            <div style='background: #EB0A1E22; border: 2px solid #EB0A1E; 
                        border-radius: 8px; padding: 16px; margin: 8px 0;'>
                <h4 style='margin: 0; color: #EB0A1E;'>Prediction Confidence</h4>
                <p style='margin: 8px 0 0 0; font-size: 24px;'>{confidence:.1%}</p>
            </div>
        """, unsafe_allow_html=True)

# Recent telemetry history
st.markdown("---")
st.subheader("📈 Recent Telemetry (Last 50 messages)")

history = buffer.get_history(selected_session, selected_car, window=50)

if len(history) > 5:
    times = list(range(len(history)))
    speeds = [msg.get("dynamics", {}).get("speed_kmh", 0) for msg in history]
    rpms = [msg.get("dynamics", {}).get("rpm", 0) for msg in history]
    throttles = [msg.get("dynamics", {}).get("throttle_pedal_pct", 0) for msg in history]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Speed (km/h)", "RPM", "Throttle (%)"),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=times, y=speeds, mode='lines', name='Speed',
                  line=dict(color='#EB0A1E', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=times, y=rpms, mode='lines', name='RPM',
                  line=dict(color='#00FF00', width=2)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=times, y=throttles, mode='lines', name='Throttle',
                  line=dict(color='#FFA500', width=2)),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Message #", row=3, col=1)
    fig.update_layout(
        height=600,
        showlegend=False,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1E1E1E",
        font=dict(color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer with last update time
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"Session: {selected_session}")

with col2:
    st.caption(f"Last update: {latest.get('timestamp_utc', 'N/A')}")

with col3:
    st.caption(f"Messages: {len(history)}")

# Auto-refresh logic
if auto_refresh:
    time.sleep(1.0 / refresh_rate)
    st.rerun()
