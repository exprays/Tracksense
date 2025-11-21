"""
Visualization utilities for race data analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from ..utils.constants import DASHBOARD, TIRE_DEGRADATION


class RaceVisualizer:
    """Create visualizations for race data"""
    
    def __init__(self):
        self.color_scheme = DASHBOARD['color_scheme']
    
    def plot_lap_times(self, df: pd.DataFrame, title: str = "Lap Time Evolution") -> go.Figure:
        """
        Plot lap times over the race
        
        Args:
            df: DataFrame with lap data
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if df.empty or 'LAP_NUMBER' not in df.columns:
            return fig
        
        # Actual lap times
        fig.add_trace(go.Scatter(
            x=df['LAP_NUMBER'],
            y=df['LAP_TIME_SECONDS'],
            mode='lines+markers',
            name='Lap Time',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        # Moving average
        if 'LAP_TIME_MA3' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'],
                y=df['LAP_TIME_MA3'],
                mode='lines',
                name='3-Lap Average',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
        
        # Best lap line
        if df['LAP_TIME_SECONDS'].notna().any():
            best_lap = df['LAP_TIME_SECONDS'].min()
            fig.add_hline(
                y=best_lap,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Best: {best_lap:.3f}s"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (seconds)",
            hovermode='x unified',
            height=DASHBOARD['chart_height'],
            template='plotly_white'
        )
        
        return fig
    
    def plot_tire_degradation(self, df: pd.DataFrame, 
                             title: str = "Tire Degradation Analysis") -> go.Figure:
        """
        Plot tire wear indicators
        
        Args:
            df: DataFrame with tire data
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Tire Life Estimate', 'Lap Time Delta from Best'),
            vertical_spacing=0.15
        )
        
        if df.empty or 'LAP_NUMBER' not in df.columns:
            return fig
        
        # Tire life estimate
        if 'TIRE_LIFE_ESTIMATE' in df.columns:
            colors = df['TIRE_LIFE_ESTIMATE'].apply(
                lambda x: self.color_scheme['tire_good'] if x > 0.75
                else self.color_scheme['tire_warning'] if x > 0.65
                else self.color_scheme['tire_critical']
            )
            
            fig.add_trace(go.Bar(
                x=df['LAP_NUMBER'],
                y=df['TIRE_LIFE_ESTIMATE'] * 100,
                name='Tire Life %',
                marker_color=colors,
                showlegend=False
            ), row=1, col=1)
            
            # Warning threshold lines
            x_range = [df['LAP_NUMBER'].min(), df['LAP_NUMBER'].max()]
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[75, 75],
                mode='lines',
                line=dict(color='orange', dash='dash', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[65, 65],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)
        
        # Delta from best lap
        if 'DELTA_FROM_BEST' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'],
                y=df['DELTA_FROM_BEST'],
                mode='lines+markers',
                name='Delta',
                line=dict(color='#d62728', width=2),
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.2)'
            ), row=2, col=1)
        
        fig.update_xaxes(title_text="Lap Number", row=2, col=1)
        fig.update_yaxes(title_text="Tire Life (%)", row=1, col=1)
        fig.update_yaxes(title_text="Delta (seconds)", row=2, col=1)
        
        fig.update_layout(
            title_text=title,
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def plot_fuel_consumption(self, df: pd.DataFrame,
                             title: str = "Fuel Management") -> go.Figure:
        """
        Plot fuel usage and remaining fuel
        
        Args:
            df: DataFrame with fuel data
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Fuel Remaining', 'Laps of Fuel'),
            specs=[[{"type": "indicator"}, {"type": "scatter"}]]
        )
        
        if df.empty or 'LAP_NUMBER' not in df.columns:
            return fig
        
        current_data = df.iloc[-1]
        
        # Fuel remaining gauge
        if 'FUEL_REMAINING' in df.columns:
            fuel_remaining = current_data['FUEL_REMAINING']
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=fuel_remaining,
                title={'text': "Fuel (Liters)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 50]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 10], 'color': self.color_scheme['fuel_critical']},
                        {'range': [10, 20], 'color': self.color_scheme['fuel_warning']},
                        {'range': [20, 50], 'color': self.color_scheme['fuel_ok']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 15
                    }
                }
            ), row=1, col=1)
        
        # Laps of fuel remaining
        if 'LAPS_OF_FUEL' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'],
                y=df['LAPS_OF_FUEL'],
                mode='lines+markers',
                name='Laps of Fuel',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=8)
            ), row=1, col=2)
            
            # Warning lines - add as separate scatter traces
            fig.add_trace(go.Scatter(
                x=[df['LAP_NUMBER'].min(), df['LAP_NUMBER'].max()],
                y=[4, 4],
                mode='lines',
                line=dict(color='orange', dash='dash', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=[df['LAP_NUMBER'].min(), df['LAP_NUMBER'].max()],
                y=[2, 2],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=2)
        
        fig.update_xaxes(title_text="Lap Number", row=1, col=2)
        fig.update_yaxes(title_text="Laps Remaining", row=1, col=2)
        
        fig.update_layout(
            title_text=title,
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def plot_sector_comparison(self, df: pd.DataFrame,
                              title: str = "Sector Time Analysis") -> go.Figure:
        """
        Compare sector times across laps
        
        Args:
            df: DataFrame with sector data
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if df.empty or not all(col in df.columns for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']):
            return fig
        
        # Sector 1
        fig.add_trace(go.Scatter(
            x=df['LAP_NUMBER'],
            y=df['S1_SECONDS'],
            mode='lines+markers',
            name='Sector 1',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Sector 2
        fig.add_trace(go.Scatter(
            x=df['LAP_NUMBER'],
            y=df['S2_SECONDS'],
            mode='lines+markers',
            name='Sector 2',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        # Sector 3
        fig.add_trace(go.Scatter(
            x=df['LAP_NUMBER'],
            y=df['S3_SECONDS'],
            mode='lines+markers',
            name='Sector 3',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Lap Number",
            yaxis_title="Sector Time (seconds)",
            hovermode='x unified',
            height=DASHBOARD['chart_height'],
            template='plotly_white'
        )
        
        return fig
    
    def plot_pit_recommendation(self, df: pd.DataFrame,
                               title: str = "Pit Stop Recommendation") -> go.Figure:
        """
        Visualize pit stop recommendation score
        
        Args:
            df: DataFrame with pit recommendation data
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Pit Stop Score', 'Contributing Factors'),
            specs=[[{"type": "indicator"}, {"type": "bar"}]]
        )
        
        if df.empty or 'LAP_NUMBER' not in df.columns:
            return fig
        
        current_data = df.iloc[-1]
        
        # Pit stop recommendation gauge
        if 'PIT_RECOMMENDATION_SCORE' in df.columns:
            pit_score = current_data['PIT_RECOMMENDATION_SCORE']
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=pit_score,
                title={'text': "Pit Now Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if pit_score > 70 else "orange" if pit_score > 40 else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ), row=1, col=1)
        
        # Contributing factors
        factors = {
            'Tire Wear': (1 - current_data.get('TIRE_LIFE_ESTIMATE', 1)) * 100,
            'Fuel Level': 100 - (current_data.get('LAPS_OF_FUEL', 10) * 10),
            'Lap Time': current_data.get('DELTA_PERCENT', 0),
            'Stint Length': min((current_data.get('LAPS_IN_STINT', 0) / 15) * 100, 100)
        }
        
        fig.add_trace(go.Bar(
            x=list(factors.values()),
            y=list(factors.keys()),
            orientation='h',
            marker_color=['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e'],
            showlegend=False
        ), row=1, col=2)
        
        fig.update_xaxes(title_text="Impact (%)", row=1, col=2, range=[0, 100])
        
        fig.update_layout(
            title_text=title,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def plot_consistency_analysis(self, df: pd.DataFrame,
                                  title: str = "Driver Consistency") -> go.Figure:
        """
        Analyze driver consistency
        
        Args:
            df: DataFrame with consistency data
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Consistency Score', 'Lap Time Distribution'),
            vertical_spacing=0.15
        )
        
        if df.empty or 'LAP_NUMBER' not in df.columns:
            return fig
        
        # Consistency score over time
        if 'CONSISTENCY_SCORE' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'],
                y=df['CONSISTENCY_SCORE'],
                mode='lines+markers',
                name='Consistency',
                line=dict(color='#9467bd', width=2),
                fill='tozeroy',
                fillcolor='rgba(148, 103, 189, 0.2)'
            ), row=1, col=1)
        
        # Lap time distribution
        if 'LAP_TIME_SECONDS' in df.columns:
            lap_times = df['LAP_TIME_SECONDS'].dropna()
            if len(lap_times) > 0:
                fig.add_trace(go.Histogram(
                    x=lap_times,
                    nbinsx=20,
                    name='Distribution',
                    marker_color='#8c564b',
                    showlegend=False
                ), row=2, col=1)
        
        fig.update_xaxes(title_text="Lap Number", row=1, col=1)
        fig.update_yaxes(title_text="Score (0-100)", row=1, col=1)
        fig.update_xaxes(title_text="Lap Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(
            title_text=title,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def plot_weather_impact(self, df: pd.DataFrame,
                           title: str = "Weather Conditions") -> go.Figure:
        """
        Visualize weather conditions and their impact
        
        Args:
            df: DataFrame with weather data
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Air Temperature', 'Track Temperature', 
                          'Humidity', 'Wind Speed'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        if df.empty:
            return fig
        
        # Air temperature
        if 'AIR_TEMP' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'] if 'LAP_NUMBER' in df.columns else df.index,
                y=df['AIR_TEMP'],
                mode='lines',
                name='Air Temp',
                line=dict(color='#e377c2', width=2),
                showlegend=False
            ), row=1, col=1)
        
        # Track temperature
        if 'TRACK_TEMP' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'] if 'LAP_NUMBER' in df.columns else df.index,
                y=df['TRACK_TEMP'],
                mode='lines',
                name='Track Temp',
                line=dict(color='#d62728', width=2),
                showlegend=False
            ), row=1, col=2)
        
        # Humidity
        if 'HUMIDITY' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'] if 'LAP_NUMBER' in df.columns else df.index,
                y=df['HUMIDITY'],
                mode='lines',
                name='Humidity',
                line=dict(color='#17becf', width=2),
                showlegend=False
            ), row=2, col=1)
        
        # Wind speed
        if 'WIND_SPEED' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'] if 'LAP_NUMBER' in df.columns else df.index,
                y=df['WIND_SPEED'],
                mode='lines',
                name='Wind Speed',
                line=dict(color='#bcbd22', width=2),
                showlegend=False
            ), row=2, col=2)
        
        fig.update_yaxes(title_text="°C", row=1, col=1)
        fig.update_yaxes(title_text="°C", row=1, col=2)
        fig.update_yaxes(title_text="%", row=2, col=1)
        fig.update_yaxes(title_text="km/h", row=2, col=2)
        
        fig.update_layout(
            title_text=title,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_dashboard_summary(self, state: Dict) -> go.Figure:
        """
        Create a summary dashboard with key metrics
        
        Args:
            state: Dictionary with current state metrics
        
        Returns:
            Plotly figure with summary metrics
        """
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
            ]
        )
        
        # Tire life
        tire_life = state.get('tire_life', 1.0) * 100
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=tire_life,
            title={'text': "Tire Life"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if tire_life > 75 else "orange" if tire_life > 65 else "red"}
            }
        ), row=1, col=1)
        
        # Degradation rate
        deg_rate = state.get('degradation_rate', 0)
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=deg_rate,
            number={'suffix': "s/lap", 'valueformat': ".3f"},
            title={'text': "Tire Degradation"},
            delta={'reference': 0, 'increasing': {'color': "red"}},
        ), row=1, col=2)
        
        # Fuel remaining
        fig.add_trace(go.Indicator(
            mode="number",
            value=state.get('laps_of_fuel', 0),
            number={'suffix': " laps", 'valueformat': ".1f"},
            title={'text': "Fuel Remaining"},
        ), row=1, col=3)
        
        # Pit recommendation score
        pit_score = state.get('pit_score', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=pit_score,
            title={'text': "Pit Now Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if pit_score > 70 else "orange" if pit_score > 40 else "green"}
            }
        ), row=2, col=1)
        
        # Consistency
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=state.get('consistency', 100),
            title={'text': "Consistency"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if state.get('consistency', 100) > 80 else "orange" if state.get('consistency', 100) > 60 else "red"}
            }
        ), row=2, col=2)
        
        # Laps in stint
        laps_in_stint = state.get('lap', 0)  # Simplified - actual stint tracking would be more complex
        fig.add_trace(go.Indicator(
            mode="number",
            value=laps_in_stint,
            number={'suffix': " laps"},
            title={'text': "Current Stint"},
        ), row=2, col=3)
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
