"""
AI-Powered Race Insights Generator
Provides natural language insights and predictive analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceInsightsGenerator:
    """Generate intelligent insights from race data and model predictions"""
    
    def __init__(self):
        self.insights_history = []
    
    def analyze_tire_strategy(self, 
                             current_lap: int,
                             total_laps: int,
                             tire_life: float,
                             degradation_rate: float,
                             tire_predictions: Dict) -> Dict:
        """
        Generate insights about tire strategy
        
        Args:
            current_lap: Current lap number
            total_laps: Total race laps
            tire_life: Current tire life (0-1)
            degradation_rate: Rate of degradation (s/lap)
            tire_predictions: Future tire predictions
        
        Returns:
            Dictionary with insights and recommendations
        """
        insights = []
        recommendations = []
        urgency = 'low'
        
        laps_remaining = total_laps - current_lap
        
        # Analyze current tire state
        if tire_life > 0.85:
            insights.append(f"Tires are in excellent condition ({tire_life*100:.1f}% life)")
            insights.append("Current degradation is minimal, good pace can be maintained")
        elif tire_life > 0.75:
            insights.append(f"Tires showing normal wear ({tire_life*100:.1f}% life)")
            insights.append(f"Degradation rate: +{degradation_rate:.3f}s per lap")
        elif tire_life > 0.65:
            insights.append(f"⚠️ Tires approaching critical wear ({tire_life*100:.1f}% life)")
            insights.append(f"Significant degradation: +{degradation_rate:.3f}s per lap")
            urgency = 'high'
            recommendations.append("Consider pitting within 2-3 laps")
        else:
            insights.append(f"🔴 CRITICAL: Tire life critically low ({tire_life*100:.1f}%)")
            insights.append("Severe performance degradation expected")
            urgency = 'critical'
            recommendations.append("PIT IMMEDIATELY - Risk of tire failure")
        
        # Predict if current tires can finish
        predicted_tire_at_finish = tire_life - (degradation_rate * laps_remaining * 0.01)
        
        if predicted_tire_at_finish < 0.5:
            insights.append(f"Projected tire life at finish: {max(0, predicted_tire_at_finish)*100:.1f}%")
            insights.append("⚠️ Current tires unlikely to last until race end")
            recommendations.append("Pit stop required to finish race safely")
        else:
            insights.append("Current tire set can potentially finish the race")
        
        # Analyze predicted degradation trend
        if tire_predictions.get('warnings'):
            insights.append("Future lap analysis:")
            for warning in tire_predictions['warnings'][:3]:  # Show top 3
                insights.append(f"  • {warning}")
        
        # Strategic recommendations
        if laps_remaining > 10 and tire_life < 0.75:
            recommendations.append("Early pit stop could enable push on fresh tires")
            recommendations.append("Consider undercut strategy if close to competitors")
        
        return {
            'category': 'Tire Strategy',
            'urgency': urgency,
            'insights': insights,
            'recommendations': recommendations,
            'confidence': 0.85,  # ML model confidence
            'predicted_tire_at_finish': max(0, predicted_tire_at_finish)
        }
    
    def analyze_pit_strategy(self,
                            current_lap: int,
                            total_laps: int,
                            pit_window: Dict,
                            strategies: List[Dict]) -> Dict:
        """
        Generate insights about pit stop strategy
        
        Args:
            current_lap: Current lap number
            total_laps: Total race laps
            pit_window: Pit window information
            strategies: List of strategy options
        
        Returns:
            Dictionary with insights
        """
        insights = []
        recommendations = []
        
        laps_remaining = total_laps - current_lap
        urgency = pit_window.get('urgency', 'low')
        
        # Analyze pit window
        if pit_window.get('should_pit'):
            optimal_lap = pit_window.get('optimal_lap', current_lap + 1)
            window_start = pit_window.get('window_start', optimal_lap)
            window_end = pit_window.get('window_end', optimal_lap)
            
            insights.append(f"Optimal pit window: Laps {window_start}-{window_end}")
            insights.append(f"Recommended pit lap: {optimal_lap}")
            insights.append(f"Reason: {pit_window.get('reason', 'Strategic timing')}")
            
            laps_until_pit = optimal_lap - current_lap
            if laps_until_pit <= 1:
                recommendations.append("🔴 PIT THIS LAP or NEXT LAP")
            elif laps_until_pit <= 3:
                recommendations.append(f"⚠️ Prepare to pit in {laps_until_pit} laps")
            else:
                recommendations.append(f"Pit window opens in {laps_until_pit} laps")
        else:
            insights.append("✓ No immediate pit stop required")
            insights.append(pit_window.get('reason', 'Can continue on current strategy'))
        
        # Compare strategies
        if strategies and len(strategies) > 1:
            best_strategy = min(strategies, key=lambda s: s.get('estimated_time', float('inf')))
            
            insights.append(f"\nStrategy comparison ({len(strategies)} options analyzed):")
            
            for i, strat in enumerate(strategies[:3], 1):  # Top 3
                time_delta = strat.get('estimated_time', 0) - best_strategy.get('estimated_time', 0)
                marker = "★" if i == 1 else " "
                
                insights.append(
                    f"{marker} {strat['name']}: "
                    f"+{time_delta:.2f}s vs best"
                )
            
            # Recommendation based on best strategy
            if best_strategy['name'] == 'No Stop':
                recommendations.append("Stay out - no-stop strategy is optimal")
                recommendations.append("Focus on tire management for remainder")
            elif best_strategy['name'] == 'One Stop':
                recommendations.append("Single pit stop strategy recommended")
                if best_strategy.get('pit_laps'):
                    recommendations.append(f"Optimal pit lap: {best_strategy['pit_laps'][0]}")
            else:
                recommendations.append("Two-stop strategy shows best predicted time")
        
        # Risk analysis
        if urgency == 'critical':
            insights.append("\n⚠️ CRITICAL RISK FACTORS:")
            insights.append("Delaying pit stop may result in performance loss or DNF")
        elif urgency == 'high':
            insights.append("\n⚠️ Risk: Tire or fuel situation requires attention")
        
        return {
            'category': 'Pit Strategy',
            'urgency': urgency,
            'insights': insights,
            'recommendations': recommendations,
            'confidence': 0.80
        }
    
    def analyze_driver_performance(self,
                                  processed_data: pd.DataFrame,
                                  driver_profile: Optional[Dict] = None) -> Dict:
        """
        Generate insights about driver performance and style
        
        Args:
            processed_data: Processed race data for driver
            driver_profile: Driver fingerprint/profile if available
        
        Returns:
            Dictionary with insights
        """
        insights = []
        recommendations = []
        
        if processed_data.empty:
            return {'category': 'Driver Performance', 'insights': [], 'recommendations': []}
        
        # Calculate statistics
        consistency = processed_data['CONSISTENCY_SCORE'].mean()
        avg_lap_time = processed_data['LAP_TIME_SECONDS'].mean()
        best_lap_time = processed_data['LAP_TIME_SECONDS'].min()
        
        # Consistency analysis
        if consistency > 85:
            insights.append(f"✓ Excellent consistency: {consistency:.1f}/100")
            insights.append("Driver maintaining very stable lap times")
        elif consistency > 75:
            insights.append(f"Good consistency: {consistency:.1f}/100")
            insights.append("Minor lap time variations within acceptable range")
        else:
            insights.append(f"⚠️ Inconsistent performance: {consistency:.1f}/100")
            insights.append("Significant lap time variations detected")
            recommendations.append("Focus on consistency to optimize race pace")
        
        # Pace analysis
        delta_best_to_avg = avg_lap_time - best_lap_time
        if delta_best_to_avg < 0.5:
            insights.append(f"Maintaining pace close to best lap (Δ {delta_best_to_avg:.3f}s)")
        else:
            insights.append(f"Pace dropped {delta_best_to_avg:.3f}s from best lap")
            recommendations.append("Potential for lap time improvement")
        
        # Sector analysis
        if all(col in processed_data.columns for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']):
            s1_avg = processed_data['S1_SECONDS'].mean()
            s2_avg = processed_data['S2_SECONDS'].mean()
            s3_avg = processed_data['S3_SECONDS'].mean()
            
            total = s1_avg + s2_avg + s3_avg
            s1_pct = (s1_avg / total) * 100
            s2_pct = (s2_avg / total) * 100
            s3_pct = (s3_avg / total) * 100
            
            insights.append(f"\nSector distribution: S1 {s1_pct:.1f}% | S2 {s2_pct:.1f}% | S3 {s3_pct:.1f}%")
            
            # Identify strongest/weakest sectors
            sectors = {'S1': s1_avg, 'S2': s2_avg, 'S3': s3_avg}
            best_sector = min(sectors, key=sectors.get)
            worst_sector = max(sectors, key=sectors.get)
            
            insights.append(f"Strongest sector: {best_sector}")
            recommendations.append(f"Optimization opportunity in {worst_sector}")
        
        # Driver profile insights
        if driver_profile:
            aggression = driver_profile.get('aggression', 0)
            if aggression > 0.05:
                insights.append(f"\nDriving style: Aggressive (degradation +{aggression:.3f}s/lap)")
                recommendations.append("Consider smoother inputs to preserve tires")
            else:
                insights.append(f"Driving style: Smooth (minimal tire stress)")
        
        return {
            'category': 'Driver Performance',
            'urgency': 'low',
            'insights': insights,
            'recommendations': recommendations,
            'metrics': {
                'consistency': consistency,
                'avg_lap_time': avg_lap_time,
                'best_lap_time': best_lap_time,
                'delta': delta_best_to_avg
            },
            'confidence': 0.90
        }
    
    def predict_race_outcome(self,
                           current_lap: int,
                           total_laps: int,
                           current_position: Optional[int],
                           processed_data: pd.DataFrame,
                           strategy: Dict) -> Dict:
        """
        Predict race outcome based on current data and strategy
        
        Args:
            current_lap: Current lap number
            total_laps: Total race laps
            current_position: Current race position if known
            processed_data: Processed race data
            strategy: Selected strategy
        
        Returns:
            Dictionary with predictions
        """
        insights = []
        predictions = {}
        
        laps_remaining = total_laps - current_lap
        
        # Predict finish time
        if not processed_data.empty:
            recent_pace = processed_data['LAP_TIME_SECONDS'].tail(5).mean()
            predicted_remaining_time = recent_pace * laps_remaining
            
            # Account for pit stops
            pit_stops = len(strategy.get('pit_laps', []))
            pit_time_loss = pit_stops * 30  # Assume 30s per stop
            
            total_predicted_time = predicted_remaining_time + pit_time_loss
            
            predictions['estimated_finish_time'] = total_predicted_time
            predictions['predicted_laps_remaining'] = laps_remaining
            
            insights.append(f"Predicted time to finish: {total_predicted_time:.1f}s")
            insights.append(f"Based on recent pace: {recent_pace:.3f}s/lap")
            
            if pit_stops > 0:
                insights.append(f"Includes {pit_stops} pit stop(s) (+{pit_time_loss}s)")
        
        # Success probability
        tire_health = processed_data['TIRE_LIFE_ESTIMATE'].iloc[-1] if not processed_data.empty else 1.0
        fuel_health = processed_data['LAPS_OF_FUEL'].iloc[-1] if not processed_data.empty else total_laps
        
        # Calculate success probability
        tire_ok = tire_health > 0.5 or pit_stops > 0
        fuel_ok = fuel_health > laps_remaining or pit_stops > 0
        
        if tire_ok and fuel_ok:
            success_prob = 0.95
            insights.append(f"\n✓ High probability of race completion ({success_prob*100:.0f}%)")
        elif tire_ok or fuel_ok:
            success_prob = 0.70
            insights.append(f"\n⚠️ Moderate risk factors present ({success_prob*100:.0f}% success)")
        else:
            success_prob = 0.40
            insights.append(f"\n🔴 High risk of DNF ({success_prob*100:.0f}% success)")
            insights.append("Strategy adjustment required")
        
        predictions['success_probability'] = success_prob
        
        # Position prediction (if we know current position)
        if current_position is not None:
            # Simplified position prediction
            if success_prob > 0.9 and tire_health > 0.7:
                predicted_position = current_position  # Hold position
                insights.append(f"Predicted finish position: P{predicted_position}")
            else:
                predicted_position = current_position + 1  # May lose positions
                insights.append(f"At risk of losing position (predicted: P{predicted_position})")
            
            predictions['predicted_position'] = predicted_position
        
        return {
            'category': 'Race Outcome Prediction',
            'urgency': 'low' if success_prob > 0.8 else 'high',
            'insights': insights,
            'predictions': predictions,
            'confidence': 0.75
        }
    
    def generate_comprehensive_insights(self,
                                       current_state: Dict,
                                       race_context: Dict,
                                       processed_data: pd.DataFrame,
                                       tire_predictions: Dict,
                                       pit_window: Dict,
                                       strategies: List[Dict]) -> Dict:
        """
        Generate comprehensive insights combining all analyses
        
        Args:
            current_state: Current race state
            race_context: Race context (laps, position, etc.)
            processed_data: Processed race data
            tire_predictions: Tire model predictions
            pit_window: Pit window information
            strategies: Strategy options
        
        Returns:
            Complete insights package
        """
        all_insights = {
            'timestamp': datetime.now().isoformat(),
            'current_lap': current_state.get('lap'),
            'sections': []
        }
        
        # Tire analysis
        tire_insights = self.analyze_tire_strategy(
            current_lap=current_state.get('lap', 0),
            total_laps=race_context.get('total_laps', 0),
            tire_life=current_state.get('tire_life', 1.0),
            degradation_rate=current_state.get('degradation_rate', 0),
            tire_predictions=tire_predictions
        )
        all_insights['sections'].append(tire_insights)
        
        # Pit strategy analysis
        pit_insights = self.analyze_pit_strategy(
            current_lap=current_state.get('lap', 0),
            total_laps=race_context.get('total_laps', 0),
            pit_window=pit_window,
            strategies=strategies
        )
        all_insights['sections'].append(pit_insights)
        
        # Driver performance analysis
        driver_insights = self.analyze_driver_performance(processed_data)
        all_insights['sections'].append(driver_insights)
        
        # Race outcome prediction
        outcome_insights = self.predict_race_outcome(
            current_lap=current_state.get('lap', 0),
            total_laps=race_context.get('total_laps', 0),
            current_position=race_context.get('position'),
            processed_data=processed_data,
            strategy=strategies[0] if strategies else {}
        )
        all_insights['sections'].append(outcome_insights)
        
        # Overall recommendations
        all_recommendations = []
        max_urgency = 'low'
        
        for section in all_insights['sections']:
            all_recommendations.extend(section.get('recommendations', []))
            
            section_urgency = section.get('urgency', 'low')
            if section_urgency == 'critical':
                max_urgency = 'critical'
            elif section_urgency == 'high' and max_urgency != 'critical':
                max_urgency = 'high'
        
        # Prioritize recommendations
        critical_recs = [r for r in all_recommendations if '🔴' in r or 'CRITICAL' in r.upper()]
        warning_recs = [r for r in all_recommendations if '⚠️' in r and r not in critical_recs]
        other_recs = [r for r in all_recommendations if r not in critical_recs and r not in warning_recs]
        
        all_insights['priority_recommendations'] = critical_recs + warning_recs + other_recs[:3]
        all_insights['overall_urgency'] = max_urgency
        
        return all_insights
    
    def format_insights_for_display(self, insights: Dict) -> str:
        """
        Format insights for dashboard display
        
        Args:
            insights: Insights dictionary
        
        Returns:
            Formatted string for display
        """
        output = []
        
        output.append("=" * 60)
        output.append("AI RACE INSIGHTS")
        output.append("=" * 60)
        output.append(f"Lap {insights.get('current_lap', 'N/A')}")
        output.append("")
        
        for section in insights.get('sections', []):
            output.append(f"\n{section['category']}")
            output.append("-" * 40)
            
            for insight in section.get('insights', []):
                output.append(f"  {insight}")
            
            if section.get('recommendations'):
                output.append("\n  Recommendations:")
                for rec in section['recommendations']:
                    output.append(f"    → {rec}")
            
            output.append("")
        
        if insights.get('priority_recommendations'):
            output.append("\n🎯 PRIORITY ACTIONS")
            output.append("-" * 40)
            for i, rec in enumerate(insights['priority_recommendations'][:5], 1):
                output.append(f"{i}. {rec}")
        
        output.append("\n" + "=" * 60)
        
        return "\n".join(output)
