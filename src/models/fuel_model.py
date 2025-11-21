"""
Fuel consumption calculator and predictor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from ..utils.constants import FUEL_CONSUMPTION, VEHICLE_SPECS
from ..utils.helpers import calculate_fuel_required

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuelCalculator:
    """Calculate and predict fuel consumption"""
    
    def __init__(self):
        self.base_rate = FUEL_CONSUMPTION['base_rate_liters_per_lap']
        self.tank_capacity = VEHICLE_SPECS['fuel_tank_capacity_liters']
    
    def calculate_fuel_per_lap(self, 
                               lap_times: List[float],
                               sector_times: Optional[List[Dict]] = None,
                               avg_speed: Optional[float] = None) -> float:
        """
        Calculate fuel consumption per lap based on driving style
        
        Args:
            lap_times: List of lap times
            sector_times: Optional sector time data
            avg_speed: Average speed
        
        Returns:
            Fuel consumption per lap in liters
        """
        # Base consumption
        fuel_per_lap = self.base_rate
        
        # Adjust based on pace variability (aggressive driving)
        if len(lap_times) > 3:
            lap_time_std = np.std(lap_times)
            avg_lap_time = np.mean(lap_times)
            
            # Higher variability = more aggressive = more fuel
            if avg_lap_time > 0:
                variability = lap_time_std / avg_lap_time
                if variability > 0.02:  # More than 2% variation
                    fuel_per_lap *= (1 + variability * 5)  # Penalty for inconsistent driving
        
        # Adjust based on average speed
        if avg_speed is not None:
            # Higher speed = more fuel (simplified model)
            if avg_speed > 120:  # km/h
                speed_factor = 1 + ((avg_speed - 120) / 1000)
                fuel_per_lap *= speed_factor
        
        # Clip to reasonable range
        fuel_per_lap = np.clip(fuel_per_lap, 0.8, 2.0)
        
        return fuel_per_lap
    
    def calculate_remaining_fuel(self,
                                laps_completed: int,
                                fuel_per_lap: Optional[float] = None,
                                starting_fuel: Optional[float] = None) -> Dict:
        """
        Calculate remaining fuel
        
        Args:
            laps_completed: Number of laps completed
            fuel_per_lap: Fuel consumption per lap (uses base rate if None)
            starting_fuel: Starting fuel amount (uses tank capacity if None)
        
        Returns:
            Dictionary with fuel information
        """
        if fuel_per_lap is None:
            fuel_per_lap = self.base_rate
        
        if starting_fuel is None:
            starting_fuel = self.tank_capacity
        
        fuel_used = laps_completed * fuel_per_lap
        fuel_remaining = max(0, starting_fuel - fuel_used)
        laps_of_fuel = fuel_remaining / fuel_per_lap if fuel_per_lap > 0 else 0
        
        # Determine fuel status
        if laps_of_fuel < 2:
            status = 'critical'
            warning = 'CRITICAL: Less than 2 laps of fuel remaining!'
        elif laps_of_fuel < 4:
            status = 'warning'
            warning = 'WARNING: Low fuel - less than 4 laps remaining'
        elif laps_of_fuel < 6:
            status = 'caution'
            warning = 'CAUTION: Monitor fuel levels'
        else:
            status = 'ok'
            warning = None
        
        return {
            'fuel_remaining': fuel_remaining,
            'laps_of_fuel': laps_of_fuel,
            'fuel_used': fuel_used,
            'fuel_per_lap': fuel_per_lap,
            'status': status,
            'warning': warning
        }
    
    def predict_fuel_to_finish(self,
                              current_lap: int,
                              total_laps: int,
                              fuel_remaining: float,
                              fuel_per_lap: Optional[float] = None) -> Dict:
        """
        Predict if there's enough fuel to finish
        
        Args:
            current_lap: Current lap number
            total_laps: Total laps in race
            fuel_remaining: Current fuel remaining
            fuel_per_lap: Fuel consumption rate
        
        Returns:
            Dictionary with prediction
        """
        if fuel_per_lap is None:
            fuel_per_lap = self.base_rate
        
        laps_remaining = total_laps - current_lap
        fuel_required = calculate_fuel_required(laps_remaining, fuel_per_lap)
        
        fuel_margin = fuel_remaining - fuel_required
        can_finish = fuel_margin >= 0
        
        if can_finish:
            margin_laps = fuel_margin / fuel_per_lap
            confidence = 'high' if margin_laps > 3 else 'medium' if margin_laps > 1 else 'low'
        else:
            margin_laps = 0
            confidence = 'none'
        
        return {
            'can_finish': can_finish,
            'fuel_required': fuel_required,
            'fuel_margin': fuel_margin,
            'margin_laps': margin_laps,
            'confidence': confidence,
            'laps_remaining': laps_remaining,
            'must_pit': not can_finish
        }
    
    def calculate_fuel_save_mode(self,
                                current_fuel: float,
                                laps_remaining: int,
                                target_margin_laps: float = 2.0) -> Dict:
        """
        Calculate fuel saving requirements
        
        Args:
            current_fuel: Current fuel remaining
            laps_remaining: Laps remaining in race
            target_margin_laps: Desired fuel margin in laps
        
        Returns:
            Fuel saving recommendations
        """
        # Calculate required fuel per lap to finish with margin
        target_fuel_for_finish = (laps_remaining + target_margin_laps) * self.base_rate
        
        if current_fuel >= target_fuel_for_finish:
            return {
                'fuel_save_required': False,
                'mode': 'normal',
                'target_fuel_per_lap': self.base_rate,
                'saving_percentage': 0,
                'message': 'No fuel saving required - sufficient fuel'
            }
        
        # Calculate required fuel per lap
        required_fuel_per_lap = current_fuel / laps_remaining
        
        # Calculate saving percentage needed
        saving_needed = (self.base_rate - required_fuel_per_lap) / self.base_rate
        saving_percentage = saving_needed * 100
        
        # Determine if fuel saving is possible
        max_fuel_save = 1 - FUEL_CONSUMPTION['fuel_save_factor']
        
        if saving_needed > max_fuel_save:
            return {
                'fuel_save_required': True,
                'mode': 'critical',
                'target_fuel_per_lap': required_fuel_per_lap,
                'saving_percentage': saving_percentage,
                'message': f'CRITICAL: Need to save {saving_percentage:.1f}% fuel - may not be achievable! Consider pitting.',
                'achievable': False
            }
        elif saving_needed > 0:
            return {
                'fuel_save_required': True,
                'mode': 'fuel_save',
                'target_fuel_per_lap': required_fuel_per_lap,
                'saving_percentage': saving_percentage,
                'message': f'Fuel save mode: Reduce consumption by {saving_percentage:.1f}%',
                'achievable': True
            }
        else:
            return {
                'fuel_save_required': False,
                'mode': 'normal',
                'target_fuel_per_lap': self.base_rate,
                'saving_percentage': 0,
                'message': 'Sufficient fuel available'
            }
    
    def simulate_fuel_strategy(self,
                              current_lap: int,
                              total_laps: int,
                              current_fuel: float,
                              fuel_per_lap: float,
                              pit_laps: List[int] = None) -> Dict:
        """
        Simulate fuel consumption for a race strategy
        
        Args:
            current_lap: Current lap number
            total_laps: Total race laps
            current_fuel: Current fuel remaining
            fuel_per_lap: Fuel consumption rate
            pit_laps: Laps on which to pit (refuel to full)
        
        Returns:
            Simulation results
        """
        if pit_laps is None:
            pit_laps = []
        
        fuel = current_fuel
        fuel_history = []
        
        for lap in range(current_lap, total_laps + 1):
            # Check if this is a pit lap
            if lap in pit_laps:
                fuel = self.tank_capacity  # Refuel to full
                fuel_history.append({
                    'lap': lap,
                    'fuel': fuel,
                    'event': 'pit_stop'
                })
            
            # Consume fuel
            fuel = max(0, fuel - fuel_per_lap)
            fuel_history.append({
                'lap': lap,
                'fuel': fuel,
                'event': 'lap_complete'
            })
            
            # Check if we ran out
            if fuel <= 0:
                return {
                    'success': False,
                    'ran_out_on_lap': lap,
                    'fuel_history': fuel_history,
                    'message': f'Ran out of fuel on lap {lap}'
                }
        
        return {
            'success': True,
            'final_fuel': fuel,
            'fuel_history': fuel_history,
            'message': f'Finished with {fuel:.1f}L remaining'
        }
    
    def get_fuel_warnings(self,
                         current_state: Dict,
                         race_info: Dict) -> List[Dict]:
        """
        Generate fuel-related warnings
        
        Args:
            current_state: Current race state
            race_info: Race information
        
        Returns:
            List of warning dictionaries
        """
        warnings = []
        
        fuel_remaining = current_state.get('fuel_remaining', 0)
        laps_of_fuel = current_state.get('laps_of_fuel', 0)
        current_lap = current_state.get('lap', 0)
        total_laps = race_info.get('total_laps', 27)
        laps_remaining = total_laps - current_lap
        
        # Critical fuel warning
        if laps_of_fuel < 2:
            warnings.append({
                'level': 'critical',
                'type': 'fuel',
                'message': 'CRITICAL: Less than 2 laps of fuel remaining!',
                'action': 'PIT IMMEDIATELY'
            })
        
        # Low fuel warning
        elif laps_of_fuel < 4:
            warnings.append({
                'level': 'warning',
                'type': 'fuel',
                'message': f'WARNING: Low fuel - {laps_of_fuel:.1f} laps remaining',
                'action': 'Consider pitting soon'
            })
        
        # Can't finish warning
        if laps_of_fuel < laps_remaining:
            shortfall = laps_remaining - laps_of_fuel
            warnings.append({
                'level': 'critical',
                'type': 'strategy',
                'message': f'Cannot finish race - short by {shortfall:.1f} laps of fuel',
                'action': 'MUST PIT or enable fuel save mode'
            })
        
        # Fuel save recommendation
        if laps_of_fuel < laps_remaining + 2:  # Less than 2 lap margin
            warnings.append({
                'level': 'info',
                'type': 'strategy',
                'message': 'Consider fuel saving mode',
                'action': 'Reduce throttle and maintain smooth driving'
            })
        
        return warnings
