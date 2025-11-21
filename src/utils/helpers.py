"""
Helper utility functions for the Race Strategy Optimizer
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional


def parse_lap_time(time_str: str) -> float:
    """
    Convert lap time string (MM:SS.mmm) to seconds
    
    Args:
        time_str: Lap time in format 'MM:SS.mmm' or 'SS.mmm'
    
    Returns:
        Time in seconds as float
    """
    if pd.isna(time_str):
        return np.nan
    
    time_str = str(time_str).strip()
    
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return np.nan


def format_lap_time(seconds: float) -> str:
    """
    Convert seconds to lap time string (MM:SS.mmm)
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if pd.isna(seconds) or seconds <= 0:
        return "N/A"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"


def calculate_gap(time1: float, time2: float) -> float:
    """
    Calculate time gap between two lap times
    
    Args:
        time1: First time in seconds
        time2: Second time in seconds
    
    Returns:
        Gap in seconds
    """
    return abs(time1 - time2)


def moving_average(data: List[float], window: int = 3) -> List[float]:
    """
    Calculate moving average
    
    Args:
        data: List of values
        window: Window size for moving average
    
    Returns:
        List of moving averages
    """
    if len(data) < window:
        return data
    
    return pd.Series(data).rolling(window=window, min_periods=1).mean().tolist()


def calculate_stint_length(laps: int, total_laps: int, num_stops: int = 1) -> List[int]:
    """
    Calculate optimal stint lengths for a race
    
    Args:
        laps: Total laps in race
        total_laps: Total laps available
        num_stops: Number of pit stops planned
    
    Returns:
        List of lap counts for each stint
    """
    stints = num_stops + 1
    base_stint = total_laps // stints
    remainder = total_laps % stints
    
    stint_lengths = [base_stint] * stints
    for i in range(remainder):
        stint_lengths[i] += 1
    
    return stint_lengths


def estimate_time_loss_in_pits(pit_stop_time: float = 45.0) -> float:
    """
    Estimate total time loss from pit stop (including in/out laps)
    
    Args:
        pit_stop_time: Time spent in pit box (seconds)
    
    Returns:
        Total time loss in seconds
    """
    # Pit lane speed limit + pit stop + exit
    pit_lane_time = 20  # seconds for pit lane travel
    return pit_stop_time + pit_lane_time


def calculate_fuel_required(laps_remaining: int, fuel_per_lap: float) -> float:
    """
    Calculate fuel required for remaining laps
    
    Args:
        laps_remaining: Number of laps left
        fuel_per_lap: Fuel consumption per lap (liters)
    
    Returns:
        Fuel required in liters
    """
    return laps_remaining * fuel_per_lap


def predict_finish_time(current_lap: int, total_laps: int, 
                        avg_lap_time: float, pit_stops_remaining: int = 0) -> float:
    """
    Predict race finish time
    
    Args:
        current_lap: Current lap number
        total_laps: Total laps in race
        avg_lap_time: Average lap time in seconds
        pit_stops_remaining: Number of pit stops still needed
    
    Returns:
        Estimated remaining time in seconds
    """
    laps_remaining = total_laps - current_lap
    racing_time = laps_remaining * avg_lap_time
    pit_time = pit_stops_remaining * estimate_time_loss_in_pits()
    
    return racing_time + pit_time


def normalize_vehicle_number(vehicle_id: str) -> int:
    """
    Extract vehicle number from vehicle ID
    
    Args:
        vehicle_id: Vehicle identifier (e.g., 'GR86-002-000')
    
    Returns:
        Vehicle number as integer
    """
    try:
        # Handle various formats
        if isinstance(vehicle_id, (int, float)):
            return int(vehicle_id)
        
        # Extract number from string
        parts = str(vehicle_id).split('-')
        if len(parts) > 1:
            return int(parts[1])
        
        return int(vehicle_id)
    except:
        return 0


def calculate_degradation_rate(lap_times: List[float]) -> float:
    """
    Calculate tire degradation rate from lap time progression
    
    Args:
        lap_times: List of lap times in seconds
    
    Returns:
        Degradation rate (seconds per lap)
    """
    if len(lap_times) < 2:
        return 0.0
    
    # Linear regression to find trend
    x = np.arange(len(lap_times))
    y = np.array(lap_times)
    
    # Remove outliers (more than 2 std from mean)
    mask = np.abs(y - np.mean(y)) < 2 * np.std(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return 0.0
    
    # Calculate slope
    slope = np.polyfit(x_clean, y_clean, 1)[0]
    return max(0, slope)  # Only positive degradation


def get_track_status_color(status: str) -> str:
    """
    Get color code for track status
    
    Args:
        status: Track status (GF, FCY, SC, RED)
    
    Returns:
        Color code
    """
    colors = {
        'GF': '#00ff00',    # Green flag
        'FCY': '#ffff00',   # Full course yellow
        'SC': '#ffff00',    # Safety car
        'RED': '#ff0000',   # Red flag
    }
    return colors.get(status, '#ffffff')
