"""
Race simulator for testing strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from ..utils.constants import TIRE_DEGRADATION, FUEL_CONSUMPTION, STRATEGY
from ..utils.helpers import estimate_time_loss_in_pits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceSimulator:
    """Simulate race scenarios to test strategies"""
    
    def __init__(self):
        self.base_lap_time = 100.0  # seconds
        self.pit_stop_time = 45.0
    
    def simulate_lap(self,
                    lap_number: int,
                    tire_life: float,
                    fuel_load: float,
                    base_lap_time: float,
                    weather_factor: float = 1.0) -> Dict:
        """
        Simulate a single lap
        
        Args:
            lap_number: Current lap number
            tire_life: Current tire life (0-1)
            fuel_load: Current fuel load (kg)
            base_lap_time: Base lap time in seconds
            weather_factor: Weather impact multiplier
        
        Returns:
            Lap simulation results
        """
        # Tire degradation effect
        tire_delta = (1 - tire_life) * 2.0  # Up to 2 seconds slower with worn tires
        
        # Fuel weight effect (lighter = faster)
        fuel_weight_delta = (50 - fuel_load) * 0.02  # 0.02s per liter lighter
        
        # Calculate lap time
        lap_time = base_lap_time + tire_delta - fuel_weight_delta
        lap_time *= weather_factor
        
        # Degrade tires
        new_tire_life = max(0, tire_life - TIRE_DEGRADATION['base_degradation_rate'])
        
        # Consume fuel
        new_fuel = max(0, fuel_load - FUEL_CONSUMPTION['base_rate_liters_per_lap'])
        
        return {
            'lap': lap_number,
            'lap_time': lap_time,
            'tire_life_start': tire_life,
            'tire_life_end': new_tire_life,
            'fuel_start': fuel_load,
            'fuel_end': new_fuel,
            'tire_delta': tire_delta,
            'fuel_delta': fuel_weight_delta
        }
    
    def simulate_pit_stop(self, lap_number: int) -> Dict:
        """
        Simulate a pit stop
        
        Args:
            lap_number: Lap number of pit stop
        
        Returns:
            Pit stop results
        """
        time_loss = estimate_time_loss_in_pits(self.pit_stop_time)
        
        return {
            'lap': lap_number,
            'time_loss': time_loss,
            'tire_life_after': 1.0,
            'fuel_after': 50.0
        }
    
    def simulate_race(self,
                     total_laps: int,
                     base_lap_time: float,
                     pit_strategy: List[int],
                     starting_tire_life: float = 1.0,
                     starting_fuel: float = 50.0,
                     weather_factors: Optional[List[float]] = None) -> Dict:
        """
        Simulate a complete race
        
        Args:
            total_laps: Number of laps in race
            base_lap_time: Base lap time in seconds
            pit_strategy: List of lap numbers to pit on
            starting_tire_life: Starting tire condition
            starting_fuel: Starting fuel load
            weather_factors: Optional list of weather factors per lap
        
        Returns:
            Complete race simulation
        """
        if weather_factors is None:
            weather_factors = [1.0] * total_laps
        
        tire_life = starting_tire_life
        fuel = starting_fuel
        total_time = 0
        
        lap_data = []
        pit_stops = []
        
        for lap in range(1, total_laps + 1):
            # Check if we need to pit
            if lap in pit_strategy:
                pit_result = self.simulate_pit_stop(lap)
                pit_stops.append(pit_result)
                
                tire_life = pit_result['tire_life_after']
                fuel = pit_result['fuel_after']
                total_time += pit_result['time_loss']
            
            # Simulate the lap
            weather_factor = weather_factors[lap - 1] if lap - 1 < len(weather_factors) else 1.0
            lap_result = self.simulate_lap(lap, tire_life, fuel, base_lap_time, weather_factor)
            
            lap_data.append(lap_result)
            total_time += lap_result['lap_time']
            
            # Update state
            tire_life = lap_result['tire_life_end']
            fuel = lap_result['fuel_end']
            
            # Check for failures
            if fuel <= 0:
                return {
                    'success': False,
                    'failure_type': 'fuel',
                    'failure_lap': lap,
                    'total_time': total_time,
                    'laps_completed': lap - 1,
                    'lap_data': lap_data,
                    'pit_stops': pit_stops,
                    'message': f'Ran out of fuel on lap {lap}'
                }
            
            if tire_life <= 0.3:  # Too worn to continue safely
                return {
                    'success': False,
                    'failure_type': 'tire_failure',
                    'failure_lap': lap,
                    'total_time': total_time,
                    'laps_completed': lap - 1,
                    'lap_data': lap_data,
                    'pit_stops': pit_stops,
                    'message': f'Tire failure on lap {lap}'
                }
        
        # Calculate statistics
        lap_times = [lap['lap_time'] for lap in lap_data]
        
        return {
            'success': True,
            'total_time': total_time,
            'laps_completed': total_laps,
            'lap_data': lap_data,
            'pit_stops': pit_stops,
            'average_lap_time': np.mean(lap_times),
            'fastest_lap': min(lap_times),
            'slowest_lap': max(lap_times),
            'final_tire_life': tire_life,
            'final_fuel': fuel,
            'message': 'Race completed successfully'
        }
    
    def compare_strategies(self,
                          total_laps: int,
                          base_lap_time: float,
                          strategies: Dict[str, List[int]]) -> pd.DataFrame:
        """
        Compare multiple pit strategies
        
        Args:
            total_laps: Total race laps
            base_lap_time: Base lap time
            strategies: Dictionary of strategy name -> pit laps
        
        Returns:
            DataFrame with strategy comparison
        """
        results = []
        
        for strategy_name, pit_laps in strategies.items():
            simulation = self.simulate_race(total_laps, base_lap_time, pit_laps)
            
            results.append({
                'strategy': strategy_name,
                'pit_stops': len(pit_laps),
                'pit_laps': ', '.join(map(str, pit_laps)) if pit_laps else 'None',
                'success': simulation['success'],
                'total_time': simulation['total_time'],
                'avg_lap_time': simulation.get('average_lap_time', 0),
                'fastest_lap': simulation.get('fastest_lap', 0),
                'final_tire_life': simulation.get('final_tire_life', 0),
                'final_fuel': simulation.get('final_fuel', 0),
                'message': simulation['message']
            })
        
        df = pd.DataFrame(results)
        
        # Sort by total time (fastest first) for successful strategies
        df_success = df[df['success']].sort_values('total_time')
        df_failure = df[~df['success']]
        
        return pd.concat([df_success, df_failure], ignore_index=True)
    
    def find_optimal_strategy(self,
                             total_laps: int,
                             base_lap_time: float,
                             max_stops: int = 2) -> Dict:
        """
        Find optimal pit strategy through simulation
        
        Args:
            total_laps: Total race laps
            base_lap_time: Base lap time
            max_stops: Maximum number of pit stops to consider
        
        Returns:
            Optimal strategy
        """
        best_strategy = None
        best_time = float('inf')
        
        # Try different stop configurations
        strategies_to_test = {}
        
        # No stop strategy
        strategies_to_test['No Stop'] = []
        
        # One stop strategies
        for pit_lap in range(8, total_laps - 5):
            strategies_to_test[f'1-Stop (L{pit_lap})'] = [pit_lap]
        
        # Two stop strategies (sample some)
        if max_stops >= 2 and total_laps > 20:
            for first_stop in range(8, total_laps // 2):
                for second_stop in range(first_stop + 8, total_laps - 5):
                    strategies_to_test[f'2-Stop (L{first_stop}, L{second_stop})'] = [first_stop, second_stop]
        
        logger.info(f"Testing {len(strategies_to_test)} strategies...")
        
        results = []
        for name, pit_laps in strategies_to_test.items():
            sim = self.simulate_race(total_laps, base_lap_time, pit_laps)
            
            if sim['success'] and sim['total_time'] < best_time:
                best_time = sim['total_time']
                best_strategy = {
                    'name': name,
                    'pit_laps': pit_laps,
                    'simulation': sim
                }
            
            results.append({
                'strategy': name,
                'time': sim['total_time'] if sim['success'] else None,
                'success': sim['success']
            })
        
        return {
            'optimal_strategy': best_strategy,
            'all_results': pd.DataFrame(results)
        }
    
    def simulate_real_time_scenario(self,
                                   current_lap: int,
                                   current_tire_life: float,
                                   current_fuel: float,
                                   total_laps: int,
                                   base_lap_time: float,
                                   next_pit_lap: Optional[int] = None) -> Dict:
        """
        Simulate from current race position
        
        Args:
            current_lap: Current lap number
            current_tire_life: Current tire condition
            current_fuel: Current fuel remaining
            total_laps: Total race laps
            base_lap_time: Base lap time
            next_pit_lap: Next planned pit lap (None for no pit)
        
        Returns:
            Simulation from current position
        """
        laps_remaining = total_laps - current_lap
        
        pit_strategy = [next_pit_lap] if next_pit_lap else []
        
        # Adjust simulation to start from current state
        tire_life = current_tire_life
        fuel = current_fuel
        total_time = 0
        
        lap_data = []
        
        for lap in range(current_lap + 1, total_laps + 1):
            if lap in pit_strategy:
                pit_result = self.simulate_pit_stop(lap)
                tire_life = pit_result['tire_life_after']
                fuel = pit_result['fuel_after']
                total_time += pit_result['time_loss']
            
            lap_result = self.simulate_lap(lap, tire_life, fuel, base_lap_time)
            lap_data.append(lap_result)
            total_time += lap_result['lap_time']
            
            tire_life = lap_result['tire_life_end']
            fuel = lap_result['fuel_end']
            
            if fuel <= 0:
                return {
                    'success': False,
                    'warning': f'Will run out of fuel on lap {lap}',
                    'estimated_time_to_end': total_time,
                    'laps_completed': lap - current_lap - 1
                }
        
        return {
            'success': True,
            'estimated_time_to_end': total_time,
            'laps_completed': laps_remaining,
            'final_tire_life': tire_life,
            'final_fuel': fuel,
            'lap_data': lap_data
        }
