"""
Multi-Driver Comparison Analytics
Compare performance, strategies, and characteristics across multiple drivers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiDriverComparison:
    """Compare and analyze multiple drivers simultaneously"""
    
    def __init__(self, processed_data: pd.DataFrame):
        """
        Initialize with preprocessed race data containing multiple drivers
        
        Args:
            processed_data: DataFrame with data for all drivers (must have 'NO' column)
        """
        self.data = processed_data
        self.drivers_data = {}
        
        # Split data by driver
        if 'NO' in processed_data.columns:
            for driver_id in processed_data['NO'].unique():
                driver_data = processed_data[processed_data['NO'] == driver_id].copy()
                self.drivers_data[driver_id] = driver_data
                logger.info(f"Added driver {driver_id} with {len(driver_data)} laps")
    
    def add_driver(self, driver_id: str, processed_data: pd.DataFrame):
        """
        Add a driver's data to comparison
        
        Args:
            driver_id: Unique driver identifier
            processed_data: Preprocessed race data for the driver
        """
        if not processed_data.empty:
            self.drivers_data[driver_id] = processed_data
            logger.info(f"Added driver {driver_id} with {len(processed_data)} laps")
    
    def get_performance_comparison(self, selected_drivers: Optional[List] = None) -> pd.DataFrame:
        """
        Compare key performance metrics across all drivers
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.drivers_data:
            return pd.DataFrame()
        
        # Filter to selected drivers if provided
        drivers_to_compare = selected_drivers if selected_drivers else list(self.drivers_data.keys())
        
        comparison = []
        
        for driver_id in drivers_to_compare:
            if driver_id not in self.drivers_data:
                continue
            data = self.drivers_data[driver_id]
            metrics = {
                'Driver': driver_id,
                'Total Laps': len(data),
                'Best Lap': data['LAP_TIME_SECONDS'].min() if 'LAP_TIME_SECONDS' in data.columns else np.nan,
                'Avg Lap': data['LAP_TIME_SECONDS'].mean() if 'LAP_TIME_SECONDS' in data.columns else np.nan,
                'Consistency': data['CONSISTENCY_SCORE'].mean() if 'CONSISTENCY_SCORE' in data.columns else np.nan,
                'Avg Tire Life': data['TIRE_LIFE_ESTIMATE'].mean() if 'TIRE_LIFE_ESTIMATE' in data.columns else np.nan,
                'Degradation Rate': data['DEGRADATION_RATE'].mean() if 'DEGRADATION_RATE' in data.columns else np.nan,
                'Top Speed': data['TOP_SPEED'].max() if 'TOP_SPEED' in data.columns else np.nan,
                'Avg Speed': data['KPH'].mean() if 'KPH' in data.columns else np.nan,
            }
            
            # Sector times
            if 'S1_SECONDS' in data.columns:
                metrics['Best S1'] = data['S1_SECONDS'].min()
                metrics['Avg S1'] = data['S1_SECONDS'].mean()
            if 'S2_SECONDS' in data.columns:
                metrics['Best S2'] = data['S2_SECONDS'].min()
                metrics['Avg S2'] = data['S2_SECONDS'].mean()
            if 'S3_SECONDS' in data.columns:
                metrics['Best S3'] = data['S3_SECONDS'].min()
                metrics['Avg S3'] = data['S3_SECONDS'].mean()
            
            comparison.append(metrics)
        
        df = pd.DataFrame(comparison)
        
        # Calculate deltas from best
        if 'Best Lap' in df.columns:
            best_lap = df['Best Lap'].min()
            df['Delta to Best'] = df['Best Lap'] - best_lap
        
        return df
    
    def get_lap_time_evolution(self, selected_drivers: Optional[List] = None) -> Dict[str, pd.DataFrame]:
        """
        Get lap time evolution for all drivers
        
        Args:
            selected_drivers: List of driver IDs to include (None = all)
        
        Returns:
            Dictionary mapping driver IDs to lap time DataFrames
        """
        drivers_to_compare = selected_drivers if selected_drivers else list(self.drivers_data.keys())
        evolution = {}
        
        for driver_id in drivers_to_compare:
            if driver_id not in self.drivers_data:
                continue
            data = self.drivers_data[driver_id]
            if 'LAP_NUMBER' in data.columns and 'LAP_TIME_SECONDS' in data.columns:
                evolution[driver_id] = data[['LAP_NUMBER', 'LAP_TIME_SECONDS']].copy()
        
        return evolution
    
    def get_consistency_comparison(self, selected_drivers: Optional[List] = None) -> pd.DataFrame:
        """
        Compare consistency metrics across drivers
        
        Args:
            selected_drivers: List of driver IDs to include (None = all)
        
        Returns:
            DataFrame with consistency analysis
        """
        drivers_to_compare = selected_drivers if selected_drivers else list(self.drivers_data.keys())
        consistency_data = []
        
        for driver_id in drivers_to_compare:
            if driver_id not in self.drivers_data:
                continue
            data = self.drivers_data[driver_id]
            if 'LAP_TIME_SECONDS' in data.columns:
                lap_times = data['LAP_TIME_SECONDS'].dropna()
                
                if len(lap_times) > 3:
                    std = lap_times.std()
                    cv = (std / lap_times.mean()) * 100  # Coefficient of variation
                    
                    consistency_data.append({
                        'Driver': driver_id,
                        'Std Dev': std,
                        'Coefficient of Variation': cv,
                        'Consistency Score': data['CONSISTENCY_SCORE'].mean() if 'CONSISTENCY_SCORE' in data.columns else np.nan,
                        'Min Lap': lap_times.min(),
                        'Max Lap': lap_times.max(),
                        'Range': lap_times.max() - lap_times.min()
                    })
        
        return pd.DataFrame(consistency_data)
    
    def get_sector_comparison(self, selected_drivers: Optional[List] = None) -> pd.DataFrame:
        """
        Compare sector performance across drivers
        
        Args:
            selected_drivers: List of driver IDs to include (None = all)
        
        Returns:
            DataFrame with combined sector comparison
        """
        drivers_to_compare = selected_drivers if selected_drivers else list(self.drivers_data.keys())
        sector_data = []
        
        for driver_id in drivers_to_compare:
            if driver_id not in self.drivers_data:
                continue
            data = self.drivers_data[driver_id]
            
            row = {'Driver': driver_id}
            
            for sector_col, sector_name in [('S1_SECONDS', 'S1'), ('S2_SECONDS', 'S2'), ('S3_SECONDS', 'S3')]:
                if sector_col in data.columns:
                    times = data[sector_col].dropna()
                    if len(times) > 0:
                        row[f'{sector_name} Best'] = times.min()
                        row[f'{sector_name} Avg'] = times.mean()
                    else:
                        row[f'{sector_name} Best'] = np.nan
                        row[f'{sector_name} Avg'] = np.nan
                else:
                    row[f'{sector_name} Best'] = np.nan
                    row[f'{sector_name} Avg'] = np.nan
            
            sector_data.append(row)
        
        return pd.DataFrame(sector_data)
    
    def get_tire_degradation_comparison(self, selected_drivers: Optional[List] = None) -> pd.DataFrame:
        """
        Compare tire degradation patterns across drivers
        
        Args:
            selected_drivers: List of driver IDs to include (None = all)
        
        Returns:
            DataFrame with tire degradation analysis
        """
        drivers_to_compare = selected_drivers if selected_drivers else list(self.drivers_data.keys())
        tire_data = []
        
        for driver_id in drivers_to_compare:
            if driver_id not in self.drivers_data:
                continue
            data = self.drivers_data[driver_id]
            if 'TIRE_LIFE_ESTIMATE' in data.columns and 'DEGRADATION_RATE' in data.columns:
                tire_data.append({
                    'Driver': driver_id,
                    'Starting Tire Life': data['TIRE_LIFE_ESTIMATE'].iloc[0] if len(data) > 0 else np.nan,
                    'Ending Tire Life': data['TIRE_LIFE_ESTIMATE'].iloc[-1] if len(data) > 0 else np.nan,
                    'Total Degradation': data['TIRE_LIFE_ESTIMATE'].iloc[0] - data['TIRE_LIFE_ESTIMATE'].iloc[-1] if len(data) > 0 else np.nan,
                    'Avg Degradation Rate': data['DEGRADATION_RATE'].mean(),
                    'Max Degradation Rate': data['DEGRADATION_RATE'].max(),
                    'Tire Management Score': data['TIRE_LIFE_ESTIMATE'].mean() * 100
                })
        
        return pd.DataFrame(tire_data)
    
    def get_head_to_head(self, driver1: str, driver2: str, lap: int) -> Dict:
        """
        Compare two drivers at a specific lap
        
        Args:
            driver1: First driver ID
            driver2: Second driver ID
            lap: Lap number to compare
        
        Returns:
            Comparison dictionary
        """
        if driver1 not in self.drivers_data or driver2 not in self.drivers_data:
            return {}
        
        data1 = self.drivers_data[driver1]
        data2 = self.drivers_data[driver2]
        
        # Get data for specific lap
        lap1 = data1[data1['LAP_NUMBER'] == lap]
        lap2 = data2[data2['LAP_NUMBER'] == lap]
        
        if lap1.empty or lap2.empty:
            return {}
        
        comparison = {
            'lap': lap,
            'driver1': {
                'id': driver1,
                'lap_time': lap1['LAP_TIME_SECONDS'].iloc[0] if 'LAP_TIME_SECONDS' in lap1.columns else np.nan,
                'tire_life': lap1['TIRE_LIFE_ESTIMATE'].iloc[0] if 'TIRE_LIFE_ESTIMATE' in lap1.columns else np.nan,
                'consistency': lap1['CONSISTENCY_SCORE'].iloc[0] if 'CONSISTENCY_SCORE' in lap1.columns else np.nan,
            },
            'driver2': {
                'id': driver2,
                'lap_time': lap2['LAP_TIME_SECONDS'].iloc[0] if 'LAP_TIME_SECONDS' in lap2.columns else np.nan,
                'tire_life': lap2['TIRE_LIFE_ESTIMATE'].iloc[0] if 'TIRE_LIFE_ESTIMATE' in lap2.columns else np.nan,
                'consistency': lap2['CONSISTENCY_SCORE'].iloc[0] if 'CONSISTENCY_SCORE' in lap2.columns else np.nan,
            }
        }
        
        # Calculate deltas
        comparison['delta_lap_time'] = comparison['driver1']['lap_time'] - comparison['driver2']['lap_time']
        comparison['delta_tire_life'] = comparison['driver1']['tire_life'] - comparison['driver2']['tire_life']
        
        return comparison
    
    def get_strategy_comparison(self) -> pd.DataFrame:
        """
        Compare pit strategy patterns across drivers
        
        Returns:
            DataFrame with strategy comparison
        """
        strategy_data = []
        
        for driver_id, data in self.drivers_data.items():
            if 'PIT_RECOMMENDATION_SCORE' in data.columns:
                strategy_data.append({
                    'Driver': driver_id,
                    'Avg Pit Score': data['PIT_RECOMMENDATION_SCORE'].mean(),
                    'Max Pit Score': data['PIT_RECOMMENDATION_SCORE'].max(),
                    'High Urgency Laps': (data['PIT_RECOMMENDATION_SCORE'] > 70).sum(),
                    'Optimal Window Laps': data['IN_OPTIMAL_PIT_WINDOW'].sum() if 'IN_OPTIMAL_PIT_WINDOW' in data.columns else 0
                })
        
        return pd.DataFrame(strategy_data)
    
    def get_ranking(self, metric: str = 'Best Lap') -> List[Tuple[str, float]]:
        """
        Get driver ranking by specific metric
        
        Args:
            metric: Metric to rank by
        
        Returns:
            List of (driver_id, metric_value) tuples sorted by rank
        """
        comparison = self.get_performance_comparison()
        
        if comparison.empty or metric not in comparison.columns:
            return []
        
        # Sort by metric (ascending for times, descending for scores)
        ascending = 'time' in metric.lower() or 'lap' in metric.lower() or 'degradation' in metric.lower()
        
        sorted_df = comparison.sort_values(metric, ascending=ascending)
        
        return list(zip(sorted_df['Driver'], sorted_df[metric]))
    
    def plot_lap_time_evolution(self, selected_drivers: Optional[List] = None):
        """
        Plot lap time evolution for selected drivers
        
        Args:
            selected_drivers: List of driver IDs to plot (None = all)
        
        Returns:
            Plotly figure
        """
        import plotly.graph_objects as go
        
        drivers_to_plot = selected_drivers if selected_drivers else list(self.drivers_data.keys())
        
        fig = go.Figure()
        
        for driver_id in drivers_to_plot:
            if driver_id not in self.drivers_data:
                continue
            data = self.drivers_data[driver_id]
            
            if 'LAP_NUMBER' in data.columns and 'LAP_TIME_SECONDS' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['LAP_NUMBER'],
                    y=data['LAP_TIME_SECONDS'],
                    mode='lines+markers',
                    name=f'Car #{driver_id}',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title="Lap Time Evolution Comparison",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (seconds)",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_tire_degradation_comparison(self, selected_drivers: Optional[List] = None):
        """
        Plot tire degradation comparison
        
        Args:
            selected_drivers: List of driver IDs to plot (None = all)
        
        Returns:
            Plotly figure
        """
        import plotly.graph_objects as go
        
        drivers_to_plot = selected_drivers if selected_drivers else list(self.drivers_data.keys())
        
        fig = go.Figure()
        
        for driver_id in drivers_to_plot:
            if driver_id not in self.drivers_data:
                continue
            data = self.drivers_data[driver_id]
            
            if 'LAP_NUMBER' in data.columns and 'TIRE_LIFE_ESTIMATE' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['LAP_NUMBER'],
                    y=data['TIRE_LIFE_ESTIMATE'] * 100,
                    mode='lines+markers',
                    name=f'Car #{driver_id}',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        # Add critical threshold line
        fig.add_hline(
            y=70, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Critical Threshold (70%)"
        )
        
        fig.update_layout(
            title="Tire Life Comparison",
            xaxis_title="Lap Number",
            yaxis_title="Tire Life (%)",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def generate_comparison_summary(self, selected_drivers: Optional[List] = None) -> str:
        """
        Generate a text summary of the comparison
        
        Args:
            selected_drivers: List of driver IDs to include (None = all)
        
        Returns:
            Formatted summary string
        """
        if not self.drivers_data:
            return "No drivers to compare"
        
        drivers_to_compare = selected_drivers if selected_drivers else list(self.drivers_data.keys())
        
        summary = []
        summary.append(f"=== Multi-Driver Comparison ({len(drivers_to_compare)} drivers) ===\n")
        
        # Performance ranking
        perf = self.get_performance_comparison(drivers_to_compare)
        if not perf.empty and 'Best Lap' in perf.columns:
            sorted_perf = perf.sort_values('Best Lap')
            summary.append("Lap Time Ranking:")
            for idx, row in sorted_perf.iterrows():
                delta = row.get('Delta to Best', 0)
                summary.append(f"  {idx+1}. Driver {row['Driver']}: {row['Best Lap']:.3f}s (+{delta:.3f}s)")
        
        summary.append("")
        
        # Consistency ranking
        consistency = self.get_consistency_comparison()
        if not consistency.empty:
            sorted_cons = consistency.sort_values('Coefficient of Variation')
            summary.append("Consistency Ranking (Lower CV = More Consistent):")
            for idx, row in sorted_cons.head(5).iterrows():
                summary.append(f"  {idx+1}. Driver {row['Driver']}: CV={row['Coefficient of Variation']:.2f}%")
        
        return "\n".join(summary)
