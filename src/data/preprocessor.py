"""
Data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from ..utils.constants import TIRE_DEGRADATION, FUEL_CONSUMPTION
from ..utils.helpers import calculate_degradation_rate, moving_average

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceDataPreprocessor:
    """Preprocess and engineer features from race data"""
    
    def __init__(self):
        self.scaler_params = {}
    
    def calculate_tire_wear(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate tire wear indicators from lap time progression
        
        Args:
            sector_data: DataFrame with sector timing data
        
        Returns:
            DataFrame with tire wear features added
        """
        df = sector_data.copy()
        
        if df.empty or 'LAP_TIME_SECONDS' not in df.columns:
            return df
        
        # Calculate lap time degradation
        df = df.sort_values('LAP_NUMBER')
        
        # Moving average lap time (3-lap window)
        df['LAP_TIME_MA3'] = df['LAP_TIME_SECONDS'].rolling(window=3, min_periods=1).mean()
        
        # Calculate delta from best lap
        if not df.empty and df['LAP_TIME_SECONDS'].notna().any():
            best_lap = df['LAP_TIME_SECONDS'].min()
            df['DELTA_FROM_BEST'] = df['LAP_TIME_SECONDS'] - best_lap
            df['DELTA_PERCENT'] = (df['DELTA_FROM_BEST'] / best_lap) * 100
        else:
            df['DELTA_FROM_BEST'] = 0
            df['DELTA_PERCENT'] = 0
        
        # Tire life estimation with progressive degradation
        # Combines stint progression with performance degradation
        if 'LAPS_IN_STINT' in df.columns:
            # Base wear from laps in stint (exponential decay)
            stint_wear = 1.0 - (1.0 - np.exp(-df['LAPS_IN_STINT'] * 0.05)) * 0.5
            
            # Additional wear from lap time degradation
            if 'DELTA_PERCENT' in df.columns and df['DELTA_PERCENT'].notna().any():
                # Normalize delta to 0-1 range (0-10% slowdown = 0-0.3 tire wear)
                perf_wear_factor = (df['DELTA_PERCENT'] / 100).clip(0, 0.3)
                df['TIRE_LIFE_ESTIMATE'] = stint_wear * (1.0 - perf_wear_factor)
            else:
                df['TIRE_LIFE_ESTIMATE'] = stint_wear
        else:
            # Fallback: progressive degradation based on lap number
            # Not linear - tires degrade faster as they wear
            lap_factor = df['LAP_NUMBER'] / df['LAP_NUMBER'].max() if len(df) > 0 else 0
            df['TIRE_LIFE_ESTIMATE'] = 1.0 - (lap_factor ** 1.5) * 0.6  # Max 60% wear
        
        df['TIRE_LIFE_ESTIMATE'] = df['TIRE_LIFE_ESTIMATE'].clip(lower=0.3, upper=1.0)  # Minimum 30% life
        
        # Degradation rate (seconds per lap) - calculate rolling rate
        df['DEGRADATION_RATE'] = 0.0
        lap_times = df['LAP_TIME_SECONDS'].dropna().values
        if len(lap_times) > 3:
            # Use last 5-10 laps to calculate current degradation trend
            window = min(10, len(lap_times))
            recent_times = lap_times[-window:]
            deg_rate = calculate_degradation_rate(recent_times)
            # Assign to the last row (current lap)
            df.loc[df.index[-1], 'DEGRADATION_RATE'] = deg_rate
            # For other laps, calculate rolling degradation
            for i in range(3, len(df)):
                window_times = lap_times[max(0, i-window):i+1]
                if len(window_times) >= 3:
                    df.loc[df.index[i], 'DEGRADATION_RATE'] = calculate_degradation_rate(window_times)
        
        # Flag critical tire wear
        df['TIRE_WARNING'] = df['TIRE_LIFE_ESTIMATE'] < TIRE_DEGRADATION['wear_threshold']
        
        return df
    
    def calculate_fuel_usage(self, sector_data: pd.DataFrame, 
                            fuel_per_lap: float = FUEL_CONSUMPTION['base_rate_liters_per_lap']) -> pd.DataFrame:
        """
        Calculate fuel usage and remaining fuel
        
        Args:
            sector_data: DataFrame with sector timing data
            fuel_per_lap: Fuel consumption per lap
        
        Returns:
            DataFrame with fuel features added
        """
        df = sector_data.copy()
        
        if df.empty:
            return df
        
        df = df.sort_values('LAP_NUMBER')
        
        # Calculate cumulative fuel used
        df['FUEL_USED'] = df['LAP_NUMBER'] * fuel_per_lap
        
        # Estimate remaining fuel (assuming 50L start)
        starting_fuel = 50.0
        df['FUEL_REMAINING'] = starting_fuel - df['FUEL_USED']
        df['FUEL_REMAINING'] = df['FUEL_REMAINING'].clip(lower=0)
        
        # Laps of fuel remaining
        df['LAPS_OF_FUEL'] = df['FUEL_REMAINING'] / fuel_per_lap
        
        # Fuel warnings
        df['FUEL_WARNING'] = df['LAPS_OF_FUEL'] < 4
        df['FUEL_CRITICAL'] = df['LAPS_OF_FUEL'] < 2
        
        return df
    
    def calculate_pace_analysis(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze pace and consistency
        
        Args:
            sector_data: DataFrame with sector timing data
        
        Returns:
            DataFrame with pace features added
        """
        df = sector_data.copy()
        
        if df.empty or 'LAP_TIME_SECONDS' not in df.columns:
            return df
        
        # Calculate rolling statistics
        df['LAP_TIME_ROLLING_MEAN'] = df['LAP_TIME_SECONDS'].rolling(window=5, min_periods=1).mean()
        df['LAP_TIME_ROLLING_STD'] = df['LAP_TIME_SECONDS'].rolling(window=5, min_periods=1).std()
        
        # Consistency score (lower std = more consistent)
        if df['LAP_TIME_ROLLING_STD'].notna().any():
            df['CONSISTENCY_SCORE'] = 100 - (df['LAP_TIME_ROLLING_STD'] * 10)
            df['CONSISTENCY_SCORE'] = df['CONSISTENCY_SCORE'].clip(lower=0, upper=100)
        else:
            df['CONSISTENCY_SCORE'] = 100
        
        # Identify outlier laps
        if df['LAP_TIME_SECONDS'].notna().sum() > 5:
            mean_time = df['LAP_TIME_SECONDS'].mean()
            std_time = df['LAP_TIME_SECONDS'].std()
            df['IS_OUTLIER'] = np.abs(df['LAP_TIME_SECONDS'] - mean_time) > (2 * std_time)
        else:
            df['IS_OUTLIER'] = False
        
        # Sector balance
        if all(col in df.columns for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']):
            total_sector = df['S1_SECONDS'] + df['S2_SECONDS'] + df['S3_SECONDS']
            df['S1_PERCENT'] = (df['S1_SECONDS'] / total_sector * 100).fillna(0)
            df['S2_PERCENT'] = (df['S2_SECONDS'] / total_sector * 100).fillna(0)
            df['S3_PERCENT'] = (df['S3_SECONDS'] / total_sector * 100).fillna(0)
        
        return df
    
    def merge_weather_data(self, sector_data: pd.DataFrame, 
                          weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weather data with sector data
        
        Args:
            sector_data: DataFrame with sector timing data
            weather_data: DataFrame with weather data
        
        Returns:
            Merged DataFrame
        """
        if sector_data.empty or weather_data.empty:
            return sector_data
        
        # For simplicity, we'll use average weather conditions for each lap
        # In real-time, we'd match by timestamp
        avg_weather = weather_data[['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 
                                    'WIND_SPEED', 'RAIN']].mean()
        
        for col in avg_weather.index:
            sector_data[col] = avg_weather[col]
        
        return sector_data
    
    def create_pit_stop_features(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to pit stop decisions
        
        Args:
            sector_data: DataFrame with sector timing data
        
        Returns:
            DataFrame with pit stop features added
        """
        df = sector_data.copy()
        
        if df.empty:
            return df
        
        df = df.sort_values('LAP_NUMBER')
        
        # Laps since start (stint length so far)
        df['LAPS_IN_STINT'] = df['LAP_NUMBER']
        
        # Optimal pit window indicator
        df['IN_OPTIMAL_PIT_WINDOW'] = (df['LAPS_IN_STINT'] >= 8) & (df['LAPS_IN_STINT'] <= 12)
        
        # Time loss from tire degradation vs fresh tires
        if 'DELTA_FROM_BEST' in df.columns:
            df['TIME_LOSS_VS_FRESH'] = df['DELTA_FROM_BEST']
        
        # Pit stop recommendation score (0-100)
        pit_score = 0
        
        if 'TIRE_LIFE_ESTIMATE' in df.columns:
            # Lower tire life = higher pit score
            pit_score += (1 - df['TIRE_LIFE_ESTIMATE']) * 50
        
        if 'LAPS_OF_FUEL' in df.columns:
            # Low fuel = higher pit score
            fuel_urgency = np.where(df['LAPS_OF_FUEL'] < 5, 
                                   (5 - df['LAPS_OF_FUEL']) * 10, 0)
            pit_score += fuel_urgency
        
        df['PIT_RECOMMENDATION_SCORE'] = np.clip(pit_score, 0, 100)
        
        return df
    
    def process_driver_data(self, driver_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for driver data
        
        Args:
            driver_data: Dictionary with driver's sector and weather data
        
        Returns:
            Fully processed DataFrame
        """
        sector_data = driver_data.get('sectors', pd.DataFrame())
        weather_data = driver_data.get('weather', pd.DataFrame())
        
        if sector_data.empty:
            logger.warning("Empty sector data provided")
            return pd.DataFrame()
        
        # Apply all preprocessing steps
        df = sector_data.copy()
        df = self.calculate_tire_wear(df)
        df = self.calculate_fuel_usage(df)
        df = self.calculate_pace_analysis(df)
        df = self.merge_weather_data(df, weather_data)
        df = self.create_pit_stop_features(df)
        
        logger.info(f"Processed {len(df)} laps with {len(df.columns)} features")
        
        return df
    
    def get_current_state(self, processed_data: pd.DataFrame, 
                         current_lap: Optional[int] = None) -> Dict:
        """
        Get current race state for a driver
        
        Args:
            processed_data: Fully processed DataFrame
            current_lap: Current lap number (uses latest if None)
        
        Returns:
            Dictionary with current state metrics
        """
        if processed_data.empty:
            return {}
        
        if current_lap is None:
            current_lap = processed_data['LAP_NUMBER'].max()
        
        current_data = processed_data[processed_data['LAP_NUMBER'] <= current_lap].iloc[-1]
        
        state = {
            'lap': int(current_data['LAP_NUMBER']),
            'last_lap_time': float(current_data.get('LAP_TIME_SECONDS', 0)),
            'tire_life': float(current_data.get('TIRE_LIFE_ESTIMATE', 1.0)),
            'fuel_remaining': float(current_data.get('FUEL_REMAINING', 0)),
            'laps_of_fuel': float(current_data.get('LAPS_OF_FUEL', 0)),
            'pit_score': float(current_data.get('PIT_RECOMMENDATION_SCORE', 0)),
            'consistency': float(current_data.get('CONSISTENCY_SCORE', 100)),
            'degradation_rate': float(current_data.get('DEGRADATION_RATE', 0)),
            'tire_warning': bool(current_data.get('TIRE_WARNING', False)),
            'fuel_warning': bool(current_data.get('FUEL_WARNING', False)),
        }
        
        return state
