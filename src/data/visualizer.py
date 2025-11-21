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
        self.template = "plotly_dark"
        self.toyota_red = "#EB0A1E"
        self.toyota_black = "#000000"
        self.toyota_gray = "#555555"

    def _apply_theme(self, fig: go.Figure, title: str = None, xaxis_title: str = None, yaxis_title: str = None):
        """Apply consistent dark theme to all plots"""
        fig.update_layout(
            template=self.template,
            title=dict(text=title, font=dict(size=18, color="#FAFAFA")),
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Helvetica Neue, sans-serif", color="#FAFAFA"),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor="#1E1E1E",
                bordercolor="#3E404D",
                font_size=12,
                font_family="Helvetica Neue, sans-serif"
            ),
            height=DASHBOARD['chart_height'],
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.update_xaxes(
            showgrid=False, 
            gridwidth=1, 
            gridcolor='#333333',
            zeroline=False,
            showspikes=True,
            spikethickness=1,
            spikecolor="#666666",
            spikemode="across",
            spikesnap="cursor"
        )
        fig.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='#333333',
            zeroline=False
        )
        return fig
    
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
        
        # Actual lap times - Area Chart Style
        fig.add_trace(go.Scatter(
            x=df['LAP_NUMBER'],
            y=df['LAP_TIME_SECONDS'],
            mode='lines',
            name='Lap Time',
            line=dict(color='#EB0A1E', width=3),
            fill='tozeroy',
            fillcolor='rgba(235, 10, 30, 0.1)', # Toyota Red with low opacity
            line_shape='spline', # Natural curve
            hovertemplate='<b>Lap %{x}</b><br>Time: %{y:.3f}s<extra></extra>'
        ))
        
        # Moving average
        if 'LAP_TIME_MA3' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'],
                y=df['LAP_TIME_MA3'],
                mode='lines',
                name='3-Lap Avg',
                line=dict(color='#FAFAFA', width=2, dash='dash'),
                hovertemplate='<b>3-Lap Avg</b><br>Time: %{y:.3f}s<extra></extra>'
            ))
        
        # Best lap line
        if df['LAP_TIME_SECONDS'].notna().any():
            best_lap = df['LAP_TIME_SECONDS'].min()
            fig.add_hline(
                y=best_lap,
                line_dash="dot",
                line_color="#A0A0A0",
                annotation_text=f"Best: {best_lap:.3f}s",
                annotation_position="bottom right",
                annotation_font_color="#A0A0A0"
            )
        
        self._apply_theme(fig, title=title, xaxis_title="Lap Number", yaxis_title="Lap Time (seconds)")
        
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
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        if df.empty or 'LAP_NUMBER' not in df.columns:
            return fig
        
        # Tire life estimate - Area Chart Style
        if 'TIRE_LIFE_ESTIMATE' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'],
                y=df['TIRE_LIFE_ESTIMATE'] * 100,
                name='Tire Life %',
                mode='lines',
                line=dict(color='#EB0A1E', width=3),
                fill='tozeroy',
                fillcolor='rgba(235, 10, 30, 0.1)',
                line_shape='spline',
                hovertemplate='<b>Lap %{x}</b><br>Life: %{y:.1f}%<extra></extra>'
            ), row=1, col=1)
            
            # Warning threshold lines
            x_range = [df['LAP_NUMBER'].min(), df['LAP_NUMBER'].max()]
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[75, 75],
                mode='lines',
                line=dict(color='#FFA500', dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip',
                name='Warning'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[65, 65],
                mode='lines',
                line=dict(color='#FF0000', dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip',
                name='Critical'
            ), row=1, col=1)
        
        # Delta from best lap
        if 'LAP_TIME_SECONDS' in df.columns:
            best_lap = df['LAP_TIME_SECONDS'].min()
            delta = df['LAP_TIME_SECONDS'] - best_lap
            
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'],
                y=delta,
                mode='lines',
                name='Delta to Best',
                fill='tozeroy',
                fillcolor='rgba(250, 250, 250, 0.1)',
                line=dict(color='#FAFAFA', width=2),
                line_shape='spline',
                hovertemplate='<b>Lap %{x}</b><br>Delta: +%{y:.3f}s<extra></extra>'
            ), row=2, col=1)
            
        self._apply_theme(fig, title=title)
        fig.update_yaxes(title_text="Tire Life %", row=1, col=1, showgrid=True, gridcolor='#333333')
        fig.update_yaxes(title_text="Delta (s)", row=2, col=1, showgrid=True, gridcolor='#333333')
        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_xaxes(title_text="Lap Number", row=2, col=1, showgrid=False)
        
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
                title={'text': "Fuel (Liters)", 'font': {'color': '#FAFAFA'}},
                delta={'reference': 50, 'font': {'color': '#FAFAFA'}},
                number={'font': {'color': '#FAFAFA'}},
                gauge={
                    'axis': {'range': [0, 50], 'tickcolor': '#FAFAFA'},
                    'bar': {'color': self.toyota_red},
                    'steps': [
                        {'range': [0, 10], 'color': '#330000'},
                        {'range': [10, 20], 'color': '#660000'},
                        {'range': [20, 50], 'color': '#1E1E1E'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 15
                    },
                    'bgcolor': "rgba(0,0,0,0)",
                    'bordercolor': "#333"
                }
            ), row=1, col=1)
        
        # Laps of fuel remaining
        if 'LAPS_OF_FUEL' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['LAP_NUMBER'],
                y=df['LAPS_OF_FUEL'],
                mode='lines+markers',
                name='Laps of Fuel',
                line=dict(color='#00FF00', width=3),
                marker=dict(size=8)
            ), row=1, col=2)
            
            # Warning lines - add as separate scatter traces
            fig.add_trace(go.Scatter(
                x=[df['LAP_NUMBER'].min(), df['LAP_NUMBER'].max()],
                y=[4, 4],
                mode='lines',
                line=dict(color='#FFA500', dash='dash', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=[df['LAP_NUMBER'].min(), df['LAP_NUMBER'].max()],
                y=[2, 2],
                mode='lines',
                line=dict(color='#FF0000', dash='dash', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=2)
        
        self._apply_theme(fig, title=title)
        fig.update_xaxes(title_text="Lap Number", row=1, col=2)
        fig.update_yaxes(title_text="Laps Remaining", row=1, col=2)
        
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
        
        self._apply_theme(fig, title=title, xaxis_title="Lap Number", yaxis_title="Sector Time (seconds)")
        
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
                title={'text': "Pit Now Score", 'font': {'color': '#FAFAFA'}},
                number={'font': {'color': '#FAFAFA'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#FAFAFA'},
                    'bar': {'color': "#FF0000" if pit_score > 70 else "#FFA500" if pit_score > 40 else "#00FF00"},
                    'steps': [
                        {'range': [0, 40], 'color': '#003300'},
                        {'range': [40, 70], 'color': '#333300'},
                        {'range': [70, 100], 'color': '#330000'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    },
                    'bgcolor': "rgba(0,0,0,0)",
                    'bordercolor': "#333"
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
            marker_color=['#FF0000', '#00FF00', '#0000FF', '#FFA500'],
            showlegend=False
        ), row=1, col=2)
        
        fig.update_xaxes(title_text="Impact (%)", row=1, col=2, range=[0, 100])
        
        self._apply_theme(fig, title=title)
        
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
                line=dict(color='#FAFAFA', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 255, 255, 0.1)'
            ), row=1, col=1)
        
        # Lap time distribution
        if 'LAP_TIME_SECONDS' in df.columns:
            lap_times = df['LAP_TIME_SECONDS'].dropna()
            if len(lap_times) > 0:
                fig.add_trace(go.Histogram(
                    x=lap_times,
                    nbinsx=20,
                    name='Distribution',
                    marker_color=self.toyota_red,
                    showlegend=False
                ), row=2, col=1)
        
        fig.update_xaxes(title_text="Lap Number", row=1, col=1)
        fig.update_yaxes(title_text="Score (0-100)", row=1, col=1)
        fig.update_xaxes(title_text="Lap Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        self._apply_theme(fig, title=title)
        
        return fig
    
    def plot_weather_impact(self, df: pd.DataFrame,
                           title: str = "Weather Conditions") -> go.Figure:
        """
        Visualize weather conditions and their impact using radar chart
        
        Args:
            df: DataFrame with weather data
            title: Plot title
        
        Returns:
            Plotly figure with radar chart
        """
        fig = go.Figure()
        
        if df.empty:
            return fig
        
        # Prepare weather metrics for radar chart
        weather_metrics = []
        weather_values = []
        
        # Calculate average values and normalize to 0-100 scale for radar display
        if 'AIR_TEMP' in df.columns and df['AIR_TEMP'].notna().any():
            avg_air_temp = df['AIR_TEMP'].mean()
            # Normalize temperature (assume range 15-40°C -> 0-100)
            normalized_air = ((avg_air_temp - 15) / 25) * 100
            weather_metrics.append('Air Temp')
            weather_values.append(max(0, min(100, normalized_air)))
        
        if 'TRACK_TEMP' in df.columns and df['TRACK_TEMP'].notna().any():
            avg_track_temp = df['TRACK_TEMP'].mean()
            # Normalize temperature (assume range 20-60°C -> 0-100)
            normalized_track = ((avg_track_temp - 20) / 40) * 100
            weather_metrics.append('Track Temp')
            weather_values.append(max(0, min(100, normalized_track)))
        
        if 'HUMIDITY' in df.columns and df['HUMIDITY'].notna().any():
            avg_humidity = df['HUMIDITY'].mean()
            weather_metrics.append('Humidity')
            weather_values.append(avg_humidity)
        
        if 'WIND_SPEED' in df.columns and df['WIND_SPEED'].notna().any():
            avg_wind = df['WIND_SPEED'].mean()
            # Normalize wind speed (assume range 0-30 km/h -> 0-100)
            normalized_wind = (avg_wind / 30) * 100
            weather_metrics.append('Wind Speed')
            weather_values.append(max(0, min(100, normalized_wind)))
        
        if 'PRESSURE' in df.columns and df['PRESSURE'].notna().any():
            avg_pressure = df['PRESSURE'].mean()
            # Normalize pressure (assume range 980-1020 -> 0-100)
            normalized_pressure = ((avg_pressure - 980) / 40) * 100
            weather_metrics.append('Pressure')
            weather_values.append(max(0, min(100, normalized_pressure)))
        
        # Add radar chart
        if weather_metrics and weather_values:
            fig.add_trace(go.Scatterpolar(
                r=weather_values,
                theta=weather_metrics,
                fill='toself',
                fillcolor='rgba(235, 10, 30, 0.3)',
                line=dict(color='#EB0A1E', width=2),
                marker=dict(size=8, color='#EB0A1E'),
                name='Weather Conditions',
                hovertemplate='<b>%{theta}</b><br>Value: %{r:.1f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=True,
                    ticks='',
                    gridcolor='#3E404D',
                    tickfont=dict(color='#A0A0A0', size=10)
                ),
                angularaxis=dict(
                    gridcolor='#3E404D',
                    tickfont=dict(color='#FAFAFA', size=12)
                ),
                bgcolor='#0E1117'
            ),
            showlegend=False,
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='#FAFAFA'),
            title=dict(
                text=title,
                font=dict(size=16, color='#FAFAFA'),
                x=0.5,
                xanchor='center'
            ),
            height=500
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
            title={'text': "Tire Life", 'font': {'color': '#FAFAFA'}},
            number={'font': {'color': '#FAFAFA'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#FAFAFA'},
                'bar': {'color': "#00FF00" if tire_life > 75 else "#FFA500" if tire_life > 65 else "#FF0000"},
                'bgcolor': "rgba(0,0,0,0)",
                'bordercolor': "#333"
            }
        ), row=1, col=1)
        
        # Degradation rate
        deg_rate = state.get('degradation_rate', 0)
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=deg_rate,
            number={'suffix': "s/lap", 'valueformat': ".3f", 'font': {'color': '#FAFAFA'}},
            title={'text': "Tire Degradation", 'font': {'color': '#FAFAFA'}},
            delta={'reference': 0, 'increasing': {'color': "#FF0000"}, 'decreasing': {'color': "#00FF00"}},
        ), row=1, col=2)
        
        # Fuel remaining
        fig.add_trace(go.Indicator(
            mode="number",
            value=state.get('laps_of_fuel', 0),
            number={'suffix': " laps", 'valueformat': ".1f", 'font': {'color': '#FAFAFA'}},
            title={'text': "Fuel Remaining", 'font': {'color': '#FAFAFA'}},
        ), row=1, col=3)
        
        # Pit recommendation score
        pit_score = state.get('pit_score', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=pit_score,
            title={'text': "Pit Now Score", 'font': {'color': '#FAFAFA'}},
            number={'font': {'color': '#FAFAFA'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#FAFAFA'},
                'bar': {'color': "#FF0000" if pit_score > 70 else "#FFA500" if pit_score > 40 else "#00FF00"},
                'bgcolor': "rgba(0,0,0,0)",
                'bordercolor': "#333"
            }
        ), row=2, col=1)
        
        # Consistency
        consistency = state.get('consistency', 100)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=consistency,
            title={'text': "Consistency", 'font': {'color': '#FAFAFA'}},
            number={'font': {'color': '#FAFAFA'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#FAFAFA'},
                'bar': {'color': "#00FF00" if consistency > 80 else "#FFA500" if consistency > 60 else "#FF0000"},
                'bgcolor': "rgba(0,0,0,0)",
                'bordercolor': "#333"
            }
        ), row=2, col=2)
        
        # Laps in stint
        laps_in_stint = state.get('lap', 0)  # Simplified - actual stint tracking would be more complex
        fig.add_trace(go.Indicator(
            mode="number",
            value=laps_in_stint,
            number={'suffix': " laps", 'font': {'color': '#FAFAFA'}},
            title={'text': "Current Stint", 'font': {'color': '#FAFAFA'}},
        ), row=2, col=3)
        
        self._apply_theme(fig)
        
        return fig
