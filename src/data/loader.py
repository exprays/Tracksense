"""
Data loading utilities for race datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from ..utils.helpers import parse_lap_time, normalize_vehicle_number

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceDataLoader:
    """Load and manage race data from multiple sources"""
    
    def __init__(self, base_path: str):
        """
        Initialize data loader
        
        Args:
            base_path: Base directory containing race datasets
        """
        self.base_path = Path(base_path)
        self.tracks = ['barber', 'cota', 'indianapolis', 'sebring']
        self.data_cache = {}
    
    def load_lap_times(self, track: str, race: int) -> pd.DataFrame:
        """
        Load lap time data
        
        Args:
            track: Track name ('barber' or 'cota')
            race: Race number (1 or 2)
        
        Returns:
            DataFrame with lap time data
        """
        cache_key = f"{track}_R{race}_lap_times"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
        
        try:
            if track == 'barber':
                file_path = self.base_path / 'barber-motorsports-park' / 'barber' / f'R{race}_barber_lap_time.csv'
            elif track == 'cota':
                file_path = self.base_path / 'circuit-of-the-americas' / 'COTA' / f'Race {race}' / f'R{race}_cota_lap_Time.csv'
            elif track == 'indianapolis':
                file_path = self.base_path / 'indianapolis' / 'indianapolis' / f'R{race}_indianapolis_motor_speedway_lap_time.csv'
            elif track == 'sebring':
                file_path = self.base_path / 'sebring' / 'sebring' / 'Sebring' / f'Race {race}' / f'sebring_lap_time_R{race}.csv'
            else:
                return pd.DataFrame()
            
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['vehicle_number'] = df['vehicle_number'].apply(normalize_vehicle_number)
            
            logger.info(f"Loaded {len(df)} lap time records for {track} R{race}")
            self.data_cache[cache_key] = df
            return df.copy()
        
        except Exception as e:
            logger.error(f"Error loading lap times for {track} R{race}: {e}")
            return pd.DataFrame()
    
    def load_sector_data(self, track: str, race: int) -> pd.DataFrame:
        """
        Load sector timing data
        
        Args:
            track: Track name
            race: Race number
        
        Returns:
            DataFrame with sector data
        """
        cache_key = f"{track}_R{race}_sectors"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
        
        try:
            if track == 'barber':
                file_path = self.base_path / 'barber-motorsports-park' / 'barber' / f'23_AnalysisEnduranceWithSections_Race {race}_Anonymized.CSV'
            elif track == 'cota':
                # COTA has inconsistent naming: Race 1 has no space, Race 2 has space before "Race"
                if race == 1:
                    file_path = self.base_path / 'circuit-of-the-americas' / 'COTA' / f'Race {race}' / f'23_AnalysisEnduranceWithSections_Race {race}_Anonymized.CSV'
                else:
                    file_path = self.base_path / 'circuit-of-the-americas' / 'COTA' / f'Race {race}' / f'23_AnalysisEnduranceWithSections_ Race {race}_Anonymized.CSV'
            elif track == 'indianapolis':
                file_path = self.base_path / 'indianapolis' / 'indianapolis' / f'23_AnalysisEnduranceWithSections_Race {race}.CSV'
            elif track == 'sebring':
                file_path = self.base_path / 'sebring' / 'sebring' / 'Sebring' / f'Race {race}' / f'23_AnalysisEnduranceWithSections_Race {race}_Anonymized.CSV'
            else:
                return pd.DataFrame()
            
            df = pd.read_csv(file_path, sep=';')
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Parse lap times
            df['LAP_TIME_SECONDS'] = df['LAP_TIME'].apply(parse_lap_time)
            
            # Parse sector times (already in seconds)
            numeric_cols = ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS', 'KPH', 'TOP_SPEED']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Loaded {len(df)} sector records for {track} R{race}")
            self.data_cache[cache_key] = df
            return df.copy()
        
        except Exception as e:
            logger.error(f"Error loading sector data for {track} R{race}: {e}")
            return pd.DataFrame()
    
    def load_weather_data(self, track: str, race: int) -> pd.DataFrame:
        """
        Load weather data
        
        Args:
            track: Track name
            race: Race number
        
        Returns:
            DataFrame with weather data
        """
        cache_key = f"{track}_R{race}_weather"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
        
        try:
            if track == 'barber':
                file_path = self.base_path / 'barber-motorsports-park' / 'barber' / f'26_Weather_Race {race}_Anonymized.CSV'
            elif track == 'cota':
                # COTA has inconsistent naming
                if race == 1:
                    file_path = self.base_path / 'circuit-of-the-americas' / 'COTA' / f'Race {race}' / f'26_Weather_Race {race}_Anonymized.CSV'
                else:
                    file_path = self.base_path / 'circuit-of-the-americas' / 'COTA' / f'Race {race}' / f'26_Weather_ Race {race}_Anonymized.CSV'
            elif track == 'indianapolis':
                file_path = self.base_path / 'indianapolis' / 'indianapolis' / f'26_Weather_Race {race}.CSV'
            elif track == 'sebring':
                file_path = self.base_path / 'sebring' / 'sebring' / 'Sebring' / f'Race {race}' / f'26_Weather_Race {race}_Anonymized.CSV'
            else:
                return pd.DataFrame()
            
            df = pd.read_csv(file_path, sep=';')
            df.columns = df.columns.str.strip()
            df['TIME_UTC_STR'] = pd.to_datetime(df['TIME_UTC_STR'])
            
            # Convert numeric columns
            numeric_cols = ['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'RAIN']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Loaded {len(df)} weather records for {track} R{race}")
            self.data_cache[cache_key] = df
            return df.copy()
        
        except Exception as e:
            logger.error(f"Error loading weather data for {track} R{race}: {e}")
            return pd.DataFrame()
    
    def load_race_results(self, track: str, race: int) -> pd.DataFrame:
        """
        Load race results
        
        Args:
            track: Track name
            race: Race number
        
        Returns:
            DataFrame with race results
        """
        cache_key = f"{track}_R{race}_results"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
        
        try:
            if track == 'barber':
                file_path = self.base_path / 'barber-motorsports-park' / 'barber' / f'03_Provisional Results_Race {race}_Anonymized.CSV'
            elif track == 'cota':
                # COTA has inconsistent naming
                if race == 1:
                    file_path = self.base_path / 'circuit-of-the-americas' / 'COTA' / f'Race {race}' / f'03_Provisional Results_Race {race}_Anonymized.CSV'
                else:
                    file_path = self.base_path / 'circuit-of-the-americas' / 'COTA' / f'Race {race}' / f'03_Provisional Results_ Race {race}_Anonymized.CSV'
            elif track == 'indianapolis':
                file_path = self.base_path / 'indianapolis' / 'indianapolis' / f'03_Provisional Results_Race {race}.CSV'
            elif track == 'sebring':
                file_path = self.base_path / 'sebring' / 'sebring' / 'Sebring' / f'Race {race}' / f'03_Provisional Results_Race {race}_Anonymized.CSV'
            else:
                return pd.DataFrame()
            
            df = pd.read_csv(file_path, sep=';')
            df.columns = df.columns.str.strip()
            
            # Parse times
            df['TOTAL_TIME_SECONDS'] = df['TOTAL_TIME'].apply(parse_lap_time)
            df['FL_TIME_SECONDS'] = df['FL_TIME'].apply(parse_lap_time)
            
            logger.info(f"Loaded {len(df)} race results for {track} R{race}")
            self.data_cache[cache_key] = df
            return df.copy()
        
        except Exception as e:
            logger.error(f"Error loading race results for {track} R{race}: {e}")
            return pd.DataFrame()
    
    def load_best_laps(self, track: str, race: int) -> pd.DataFrame:
        """
        Load best laps data
        
        Args:
            track: Track name
            race: Race number
        
        Returns:
            DataFrame with best laps
        """
        cache_key = f"{track}_R{race}_best_laps"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
        
        try:
            if track == 'barber':
                file_path = self.base_path / 'barber-motorsports-park' / 'barber' / f'99_Best 10 Laps By Driver_Race {race}_Anonymized.CSV'
            elif track == 'cota':
                # COTA has inconsistent naming
                if race == 1:
                    file_path = self.base_path / 'circuit-of-the-americas' / 'COTA' / f'Race {race}' / f'99_Best 10 Laps By Driver_Race {race}_Anonymized.CSV'
                else:
                    file_path = self.base_path / 'circuit-of-the-americas' / 'COTA' / f'Race {race}' / f'99_Best 10 Laps By Driver_ Race {race}_Anonymized.CSV'
            elif track == 'indianapolis':
                file_path = self.base_path / 'indianapolis' / 'indianapolis' / f'99_Best 10 Laps By Driver_Race {race}.CSV'
            elif track == 'sebring':
                file_path = self.base_path / 'sebring' / 'sebring' / 'Sebring' / f'Race {race}' / f'99_Best 10 Laps By Driver_Race {race}_Anonymized.CSV'
            else:
                return pd.DataFrame()
            
            df = pd.read_csv(file_path, sep=';')
            df.columns = df.columns.str.strip()
            
            # Parse all lap times
            lap_cols = [col for col in df.columns if col.startswith('BESTLAP_') and not col.endswith('_LAPNUM')]
            for col in lap_cols:
                df[f"{col}_SECONDS"] = df[col].apply(parse_lap_time)
            
            df['AVERAGE_SECONDS'] = df['AVERAGE'].apply(parse_lap_time)
            
            logger.info(f"Loaded {len(df)} best lap records for {track} R{race}")
            self.data_cache[cache_key] = df
            return df.copy()
        
        except Exception as e:
            logger.error(f"Error loading best laps for {track} R{race}: {e}")
            return pd.DataFrame()
    
    def get_driver_data(self, track: str, race: int, vehicle_number: int) -> Dict[str, pd.DataFrame]:
        """
        Load all data for a specific driver
        
        Args:
            track: Track name
            race: Race number
            vehicle_number: Vehicle number
        
        Returns:
            Dictionary with all data for the driver
        """
        sector_data = self.load_sector_data(track, race)
        driver_sectors = sector_data[sector_data['NUMBER'] == vehicle_number].copy()
        
        weather_data = self.load_weather_data(track, race)
        
        best_laps = self.load_best_laps(track, race)
        driver_best = best_laps[best_laps['NUMBER'] == vehicle_number].copy()
        
        return {
            'sectors': driver_sectors,
            'weather': weather_data,
            'best_laps': driver_best
        }
    
    def get_available_drivers(self, track: str, race: int) -> List[int]:
        """
        Get list of driver numbers for a race
        
        Args:
            track: Track name
            race: Race number
        
        Returns:
            List of vehicle numbers
        """
        sector_data = self.load_sector_data(track, race)
        if not sector_data.empty:
            return sorted(sector_data['NUMBER'].unique().tolist())
        return []
    
    def clear_cache(self):
        """Clear all cached data"""
        self.data_cache.clear()
        logger.info("Data cache cleared")
