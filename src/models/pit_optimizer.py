"""
Pit stop optimization algorithm
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from ..utils.constants import STRATEGY, VEHICLE_SPECS, TIRE_DEGRADATION
from ..utils.helpers import estimate_time_loss_in_pits, predict_finish_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PitStopOptimizer:
    """Optimize pit stop timing and strategy"""
    
    def __init__(self):
        self.pit_stop_time = VEHICLE_SPECS['estimated_pit_stop_time']
        self.min_laps_before_pit = STRATEGY['min_laps_before_pit']
    
    def calculate_pit_window(self, 
                            current_lap: int,
                            total_laps: int,
                            tire_life: float,
                            fuel_laps_remaining: float,
                            laps_since_pit: int = 0) -> Dict:
        """
        Calculate optimal pit stop window
        
        Args:
            current_lap: Current lap number
            total_laps: Total race laps
            tire_life: Current tire life (0-1)
            fuel_laps_remaining: Laps of fuel remaining
            laps_since_pit: Laps since last pit stop
        
        Returns:
            Dictionary with pit window information
        """
        laps_remaining = total_laps - current_lap
        
        # Can we make it to the end without pitting?
        can_finish_no_pit = (
            tire_life > TIRE_DEGRADATION['wear_threshold'] and
            fuel_laps_remaining >= laps_remaining and
            laps_since_pit < 20  # Maximum stint length
        )
        
        if can_finish_no_pit:
            return {
                'should_pit': False,
                'urgency': 'low',
                'optimal_lap': None,
                'reason': 'Can finish race without pitting',
                'window_start': None,
                'window_end': None
            }
        
        # Calculate urgency
        urgency = 'low'
        reasons = []
        
        # Tire-based urgency
        if tire_life < 0.65:
            urgency = 'critical'
            reasons.append('Critical tire wear')
        elif tire_life < 0.75:
            urgency = 'high' if urgency != 'critical' else urgency
            reasons.append('High tire wear')
        
        # Fuel-based urgency
        if fuel_laps_remaining < 3:
            urgency = 'critical'
            reasons.append('Critical fuel level')
        elif fuel_laps_remaining < 5:
            urgency = 'high' if urgency != 'critical' else urgency
            reasons.append('Low fuel')
        
        # Calculate optimal pit lap
        if urgency == 'critical':
            # Pit ASAP
            optimal_lap = current_lap + 1
            window_start = current_lap + 1
            window_end = current_lap + 2
        else:
            # Calculate based on tire degradation and strategy
            laps_until_critical_tire = int((tire_life - 0.65) / TIRE_DEGRADATION['base_degradation_rate'])
            laps_until_critical_fuel = int(fuel_laps_remaining - 3)
            
            # Take the minimum (most urgent constraint)
            max_laps_can_wait = min(laps_until_critical_tire, laps_until_critical_fuel)
            max_laps_can_wait = max(1, max_laps_can_wait)
            
            # Optimal window
            window_start = current_lap + max(1, max_laps_can_wait - 2)
            window_end = current_lap + max_laps_can_wait
            optimal_lap = current_lap + (max_laps_can_wait // 2)
            
            # Adjust for race end
            if window_end > total_laps - 3:
                window_end = total_laps - 3
                optimal_lap = min(optimal_lap, window_end)
        
        return {
            'should_pit': True,
            'urgency': urgency,
            'optimal_lap': int(optimal_lap),
            'reason': ', '.join(reasons) if reasons else 'Strategic pit stop',
            'window_start': int(window_start),
            'window_end': int(window_end)
        }
    
    def evaluate_undercut_opportunity(self,
                                     my_lap_time: float,
                                     competitor_lap_time: float,
                                     gap_seconds: float,
                                     my_tire_life: float,
                                     their_tire_life: float) -> Dict:
        """
        Evaluate undercut strategy opportunity
        
        Args:
            my_lap_time: Current lap time
            competitor_lap_time: Competitor's lap time
            gap_seconds: Gap to competitor
            my_tire_life: Our tire life
            their_tire_life: Competitor's tire life
        
        Returns:
            Dictionary with undercut analysis
        """
        # Time gain from fresh tires
        my_tire_delta = (1 - my_tire_life) * 2.0  # seconds per lap lost to tire wear
        fresh_tire_advantage = my_tire_delta
        
        # Competitor's tire disadvantage
        their_tire_delta = (1 - their_tire_life) * 2.0
        
        # Net advantage per lap with fresh tires
        net_advantage_per_lap = fresh_tire_advantage + (their_tire_delta - my_tire_delta)
        
        # Account for pit stop time loss
        pit_time_loss = estimate_time_loss_in_pits(self.pit_stop_time)
        
        # Laps needed to make up the gap
        if net_advantage_per_lap > 0:
            laps_to_undercut = (gap_seconds + pit_time_loss) / net_advantage_per_lap
        else:
            laps_to_undercut = float('inf')
        
        # Undercut advantage from strategy
        undercut_boost = STRATEGY['undercut_advantage']
        effective_laps = laps_to_undercut - (undercut_boost / net_advantage_per_lap) if net_advantage_per_lap > 0 else laps_to_undercut
        
        opportunity_score = 0
        if gap_seconds < 5 and effective_laps < 5:
            opportunity_score = 100
        elif gap_seconds < 10 and effective_laps < 8:
            opportunity_score = 70
        elif effective_laps < 12:
            opportunity_score = 40
        else:
            opportunity_score = 10
        
        return {
            'viable': effective_laps < 10,
            'opportunity_score': int(opportunity_score),
            'laps_to_overtake': int(effective_laps) if effective_laps != float('inf') else None,
            'advantage_per_lap': net_advantage_per_lap,
            'recommendation': 'Strong undercut opportunity' if opportunity_score > 70 
                            else 'Possible undercut' if opportunity_score > 40
                            else 'Undercut unlikely to succeed'
        }
    
    def simulate_strategy(self,
                         current_lap: int,
                         total_laps: int,
                         current_tire_life: float,
                         avg_lap_time: float,
                         pit_laps: List[int]) -> Dict:
        """
        Simulate a race strategy
        
        Args:
            current_lap: Current lap number
            total_laps: Total race laps
            current_tire_life: Current tire life
            avg_lap_time: Average lap time
            pit_laps: List of laps to pit on
        
        Returns:
            Strategy simulation results
        """
        laps_remaining = total_laps - current_lap
        total_time = 0
        tire_life = current_tire_life
        current_position = current_lap
        
        stints = []
        current_stint_start = current_lap
        
        for lap in range(current_lap, total_laps + 1):
            # Check if this is a pit lap
            if lap in pit_laps:
                stint_length = lap - current_stint_start
                stints.append({
                    'start_lap': current_stint_start,
                    'end_lap': lap,
                    'length': stint_length,
                    'tire_life_start': tire_life,
                    'tire_life_end': max(0, tire_life - stint_length * TIRE_DEGRADATION['base_degradation_rate'])
                })
                
                # Add pit stop time
                total_time += estimate_time_loss_in_pits(self.pit_stop_time)
                
                # Reset tire life
                tire_life = 1.0
                current_stint_start = lap + 1
            else:
                # Calculate lap time with tire degradation
                tire_delta = (1 - tire_life) * 2.0  # seconds slower per tire life lost
                lap_time = avg_lap_time + tire_delta
                total_time += lap_time
                
                # Degrade tires
                tire_life = max(0, tire_life - TIRE_DEGRADATION['base_degradation_rate'])
        
        # Final stint
        if current_stint_start <= total_laps:
            stint_length = total_laps - current_stint_start + 1
            stints.append({
                'start_lap': current_stint_start,
                'end_lap': total_laps,
                'length': stint_length,
                'tire_life_start': tire_life,
                'tire_life_end': max(0, tire_life - stint_length * TIRE_DEGRADATION['base_degradation_rate'])
            })
        
        return {
            'total_time': total_time,
            'number_of_stops': len(pit_laps),
            'stints': stints,
            'average_stint_length': np.mean([s['length'] for s in stints]) if stints else 0,
            'final_tire_life': tire_life
        }
    
    def compare_strategies(self,
                          current_lap: int,
                          total_laps: int,
                          current_tire_life: float,
                          avg_lap_time: float) -> List[Dict]:
        """
        Compare different pit stop strategies
        
        Args:
            current_lap: Current lap number
            total_laps: Total race laps
            current_tire_life: Current tire life
            avg_lap_time: Average lap time
        
        Returns:
            List of strategy options with simulations
        """
        strategies = []
        
        # Strategy 1: No stop (if viable)
        no_stop = self.simulate_strategy(current_lap, total_laps, current_tire_life, avg_lap_time, [])
        strategies.append({
            'name': 'No Stop',
            'description': 'Run to the end without pitting',
            'simulation': no_stop,
            'viable': no_stop['final_tire_life'] > 0.65
        })
        
        # Strategy 2: One stop - early
        laps_remaining = total_laps - current_lap
        if laps_remaining > 10:
            early_pit = current_lap + (laps_remaining // 3)
            early_stop = self.simulate_strategy(current_lap, total_laps, current_tire_life, avg_lap_time, [early_pit])
            strategies.append({
                'name': 'One Stop (Early)',
                'description': f'Pit on lap {early_pit}',
                'simulation': early_stop,
                'viable': True
            })
        
        # Strategy 3: One stop - late
        if laps_remaining > 10:
            late_pit = current_lap + (2 * laps_remaining // 3)
            late_stop = self.simulate_strategy(current_lap, total_laps, current_tire_life, avg_lap_time, [late_pit])
            strategies.append({
                'name': 'One Stop (Late)',
                'description': f'Pit on lap {late_pit}',
                'simulation': late_stop,
                'viable': True
            })
        
        # Strategy 4: Two stops (for longer races)
        if laps_remaining > 20:
            pit1 = current_lap + (laps_remaining // 3)
            pit2 = current_lap + (2 * laps_remaining // 3)
            two_stops = self.simulate_strategy(current_lap, total_laps, current_tire_life, avg_lap_time, [pit1, pit2])
            strategies.append({
                'name': 'Two Stops',
                'description': f'Pit on laps {pit1} and {pit2}',
                'simulation': two_stops,
                'viable': True
            })
        
        # Sort by total time (fastest first)
        viable_strategies = [s for s in strategies if s['viable']]
        viable_strategies.sort(key=lambda x: x['simulation']['total_time'])
        
        return viable_strategies
    
    def get_real_time_recommendation(self, current_state: Dict, race_info: Dict) -> Dict:
        """
        Get real-time pit stop recommendation
        
        Args:
            current_state: Current race state
            race_info: Race information (total laps, etc.)
        
        Returns:
            Pit stop recommendation
        """
        pit_window = self.calculate_pit_window(
            current_lap=current_state['lap'],
            total_laps=race_info.get('total_laps', 27),
            tire_life=current_state['tire_life'],
            fuel_laps_remaining=current_state['laps_of_fuel'],
            laps_since_pit=current_state.get('laps_in_stint', current_state['lap'])
        )
        
        strategies = self.compare_strategies(
            current_lap=current_state['lap'],
            total_laps=race_info.get('total_laps', 27),
            current_tire_life=current_state['tire_life'],
            avg_lap_time=current_state['last_lap_time']
        )
        
        return {
            'pit_window': pit_window,
            'strategies': strategies[:3],  # Top 3 strategies
            'recommended_strategy': strategies[0] if strategies else None
        }
