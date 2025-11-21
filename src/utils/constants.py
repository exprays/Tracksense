"""
Constants and configuration for the Race Strategy Optimizer
"""

# Track Information
TRACKS = {
    'barber': {
        'name': 'Barber Motorsports Park',
        'length_km': 3.7,
        'sectors': 3
    },
    'cota': {
        'name': 'Circuit of the Americas',
        'length_km': 5.513,
        'sectors': 3
    }
}

# Vehicle Specifications (Toyota GR86)
VEHICLE_SPECS = {
    'fuel_tank_capacity_liters': 50,
    'tire_compounds': ['Soft', 'Medium', 'Hard'],
    'estimated_pit_stop_time': 45,  # seconds
}

# Tire Degradation Parameters
TIRE_DEGRADATION = {
    'base_degradation_rate': 0.02,  # per lap
    'temperature_factor': 0.001,    # degradation increase per degree
    'wear_threshold': 0.7,          # 70% = critical wear
    'optimal_range': (0.9, 1.0)     # 90-100% tire life is optimal
}

# Fuel Consumption
FUEL_CONSUMPTION = {
    'base_rate_liters_per_lap': 1.2,
    'aggressive_factor': 1.15,      # multiplier for aggressive driving
    'fuel_save_factor': 0.90,       # multiplier for fuel saving mode
}

# Strategy Parameters
STRATEGY = {
    'min_laps_before_pit': 3,
    'optimal_pit_window': (8, 12),  # laps 8-12 typically optimal
    'undercut_advantage': 2.5,      # seconds gained from undercut
    'overcut_advantage': 1.8,       # seconds gained from overcut
}

# Alert Thresholds
ALERTS = {
    'tire_critical': 0.65,
    'tire_warning': 0.75,
    'fuel_critical_laps': 2,
    'fuel_warning_laps': 4,
    'gap_undercut_opportunity': 3.0,  # seconds
}

# Weather Impact
WEATHER = {
    'rain_lap_time_increase': 0.08,  # 8% slower in rain
    'temp_optimal_range': (20, 30),   # celsius
    'wind_impact_threshold': 15,      # km/h
}

# Data Columns
TELEMETRY_COLUMNS = [
    'lap', 'vehicle_number', 'timestamp', 'meta_time'
]

LAP_TIME_COLUMNS = [
    'lap', 'vehicle_number', 'timestamp', 'meta_time'
]

SECTOR_COLUMNS = [
    'NUMBER', 'LAP_NUMBER', 'LAP_TIME', 'S1', 'S2', 'S3', 
    'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS', 'KPH', 'TOP_SPEED'
]

WEATHER_COLUMNS = [
    'TIME_UTC_SECONDS', 'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 
    'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION', 'RAIN'
]

# Dashboard Configuration
DASHBOARD = {
    'refresh_interval': 5,  # seconds
    'max_laps_display': 30,
    'chart_height': 400,
    'color_scheme': {
        'tire_good': '#00ff00',
        'tire_warning': '#ffff00',
        'tire_critical': '#ff0000',
        'fuel_ok': '#00ff00',
        'fuel_warning': '#ffff00',
        'fuel_critical': '#ff0000',
    }
}
