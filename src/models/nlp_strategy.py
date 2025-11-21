"""
Natural Language Strategy Interface
Allows users to ask questions about race strategy in plain English
"""

import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyQueryEngine:
    """Natural language interface for race strategy queries"""
    
    def __init__(self):
        self.context = {}
        self.conversation_history = []
        
        # Intent patterns
        self.intent_patterns = {
            'tire_status': [
                r'tire.*life', r'how.*tire', r'tire.*condition',
                r'tire.*degradation', r'tire.*wear'
            ],
            'pit_timing': [
                r'when.*pit', r'should.*pit', r'pit.*stop',
                r'optimal.*pit', r'pit.*window'
            ],
            'fuel_status': [
                r'fuel.*remaining', r'how.*fuel', r'fuel.*left',
                r'laps.*fuel', r'run.*out.*fuel'
            ],
            'performance': [
                r'lap.*time', r'how.*fast', r'pace', r'performance',
                r'best.*lap', r'fastest.*lap'
            ],
            'comparison': [
                r'compare', r'vs', r'versus', r'difference.*between',
                r'better.*than', r'faster.*than'
            ],
            'strategy': [
                r'what.*strategy', r'best.*strategy', r'recommend',
                r'should.*i', r'what.*do'
            ],
            'weather': [
                r'weather', r'temperature', r'rain', r'wind',
                r'track.*condition'
            ],
            'consistency': [
                r'consistent', r'consistency', r'stable', r'variance',
                r'variation'
            ]
        }
    
    def set_context(self, current_state: Dict, processed_data: pd.DataFrame,
                   race_context: Dict):
        """
        Set the current race context for queries
        
        Args:
            current_state: Current race state dictionary
            processed_data: Processed race DataFrame
            race_context: Race context information
        """
        self.context = {
            'current_state': current_state,
            'data': processed_data,
            'race_context': race_context
        }
    
    def detect_intent(self, query: str) -> str:
        """
        Detect the user's intent from their query
        
        Args:
            query: User query string
        
        Returns:
            Intent category
        """
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return 'general'
    
    def answer_tire_query(self, query: str) -> str:
        """Generate answer for tire-related questions"""
        if not self.context:
            return "Please load race data first to analyze tire status."
        
        state = self.context['current_state']
        tire_life = state.get('tire_life', 0) * 100
        deg_rate = state.get('degradation_rate', 0)
        
        response = f"**Tire Status Analysis:**\n\n"
        response += f"• Current tire life: **{tire_life:.1f}%**\n"
        response += f"• Degradation rate: **+{deg_rate:.3f}s per lap**\n\n"
        
        if tire_life > 75:
            response += "✓ Tires are in **excellent condition**. No immediate concerns.\n"
        elif tire_life > 65:
            response += "⚠️ Tires showing **moderate wear**. Monitor closely and consider pitting soon.\n"
        else:
            response += "🔴 **CRITICAL**: Tire life is low. You should pit immediately to avoid performance loss.\n"
        
        # Estimate laps remaining on tires
        if deg_rate > 0:
            laps_left = (tire_life - 65) / (deg_rate * 100) if deg_rate > 0 else 999
            if laps_left > 0 and laps_left < 50:
                response += f"\nEstimated laps until critical wear: **~{int(laps_left)} laps**"
        
        return response
    
    def answer_pit_query(self, query: str) -> str:
        """Generate answer for pit stop questions"""
        if not self.context:
            return "Please load race data first to analyze pit strategy."
        
        state = self.context['current_state']
        tire_life = state.get('tire_life', 1.0)
        fuel_laps = state.get('laps_of_fuel', 0)
        current_lap = state.get('lap', 0)
        total_laps = self.context['race_context'].get('total_laps', 0)
        
        response = f"**Pit Stop Analysis:**\n\n"
        
        # Determine if should pit
        should_pit = tire_life < 0.70 or fuel_laps < 5
        laps_remaining = total_laps - current_lap
        
        if should_pit:
            reasons = []
            if tire_life < 0.70:
                reasons.append(f"tire life low ({tire_life*100:.1f}%)")
            if fuel_laps < 5:
                reasons.append(f"low fuel ({fuel_laps:.1f} laps remaining)")
            
            response += f"🔴 **YES, pit soon!**\n\n"
            response += f"**Reasons:** {', '.join(reasons)}\n\n"
            response += f"**Recommendation:** Pit within the next **1-2 laps**.\n"
        elif laps_remaining <= 5:
            response += f"✓ **No pit stop needed** - you can finish the race from here.\n"
            response += f"Only **{laps_remaining} laps** remaining.\n"
        else:
            response += f"✓ **Not yet** - current conditions are good.\n\n"
            response += f"• Tire life: {tire_life*100:.1f}% (target: pit below 70%)\n"
            response += f"• Fuel: {fuel_laps:.1f} laps (target: pit below 5 laps)\n\n"
            
            # Estimate optimal pit lap
            if tire_life > 0.75:
                est_pit_lap = current_lap + int((tire_life - 0.70) / 0.02)
                if est_pit_lap < total_laps:
                    response += f"**Estimated optimal pit lap:** Lap {est_pit_lap}\n"
        
        return response
    
    def answer_fuel_query(self, query: str) -> str:
        """Generate answer for fuel-related questions"""
        if not self.context:
            return "Please load race data first to analyze fuel status."
        
        state = self.context['current_state']
        fuel_remaining = state.get('fuel_remaining', 0)
        laps_of_fuel = state.get('laps_of_fuel', 0)
        current_lap = state.get('lap', 0)
        total_laps = self.context['race_context'].get('total_laps', 0)
        laps_remaining = total_laps - current_lap
        
        response = f"**Fuel Status Analysis:**\n\n"
        response += f"• Fuel remaining: **{fuel_remaining:.1f} liters**\n"
        response += f"• Estimated laps: **{laps_of_fuel:.1f} laps**\n"
        response += f"• Laps to finish: **{laps_remaining} laps**\n\n"
        
        if laps_of_fuel >= laps_remaining + 2:
            response += "✓ **Excellent** - you have plenty of fuel to finish the race comfortably.\n"
        elif laps_of_fuel >= laps_remaining:
            response += "✓ **Sufficient** - you can make it to the end, but with little margin.\n"
        elif laps_of_fuel >= laps_remaining - 1:
            response += "⚠️ **Tight** - consider fuel saving mode or pit for splash and dash.\n"
        else:
            response += "🔴 **CRITICAL** - you will run out of fuel! Pit stop required.\n"
            shortfall = laps_remaining - laps_of_fuel
            response += f"\nYou're short by approximately **{shortfall:.1f} laps** of fuel.\n"
        
        return response
    
    def answer_performance_query(self, query: str) -> str:
        """Generate answer for performance questions"""
        if not self.context:
            return "Please load race data first to analyze performance."
        
        data = self.context['data']
        state = self.context['current_state']
        
        if data.empty or 'LAP_TIME_SECONDS' not in data.columns:
            return "Insufficient lap time data for performance analysis."
        
        best_lap = data['LAP_TIME_SECONDS'].min()
        current_lap = state.get('last_lap_time', 0)
        avg_lap = data['LAP_TIME_SECONDS'].mean()
        consistency = state.get('consistency', 0)
        
        response = f"**Performance Analysis:**\n\n"
        response += f"• Best lap: **{best_lap:.3f}s**\n"
        response += f"• Average lap: **{avg_lap:.3f}s**\n"
        response += f"• Last lap: **{current_lap:.3f}s** (Δ +{current_lap - best_lap:.3f}s)\n"
        response += f"• Consistency score: **{consistency:.1f}/100**\n\n"
        
        if current_lap - best_lap < 0.5:
            response += "✓ **Excellent pace** - you're running close to your best lap time!\n"
        elif current_lap - best_lap < 1.0:
            response += "✓ **Good pace** - minor drop-off from best, likely due to tire wear.\n"
        else:
            response += "⚠️ **Slower pace** - significant drop-off. Check tires and adjust strategy.\n"
        
        if consistency > 85:
            response += "✓ Very **consistent** driver - minimal lap time variation.\n"
        elif consistency > 75:
            response += "Good consistency with some minor variations.\n"
        else:
            response += "⚠️ Inconsistent laps - focus on smoothness and rhythm.\n"
        
        return response
    
    def answer_strategy_query(self, query: str) -> str:
        """Generate answer for general strategy questions"""
        if not self.context:
            return "Please load race data first for strategy recommendations."
        
        state = self.context['current_state']
        current_lap = state.get('lap', 0)
        total_laps = self.context['race_context'].get('total_laps', 0)
        tire_life = state.get('tire_life', 1.0) * 100
        fuel_laps = state.get('laps_of_fuel', 0)
        laps_remaining = total_laps - current_lap
        
        response = f"**Strategy Recommendation (Lap {current_lap}/{total_laps}):**\n\n"
        
        # Analyze current situation
        can_finish_no_pit = (tire_life > 70 and fuel_laps >= laps_remaining)
        
        if can_finish_no_pit:
            response += "✓ **No-Stop Strategy Viable**\n\n"
            response += "Current conditions allow you to finish without pitting:\n"
            response += f"• Tires: {tire_life:.1f}% (sufficient)\n"
            response += f"• Fuel: {fuel_laps:.1f} laps (sufficient)\n\n"
            response += "**Recommendation:** Stay out and push for maximum track position.\n"
        else:
            response += "**One-Stop Strategy Recommended**\n\n"
            
            if tire_life < 70:
                response += f"• Tires are worn ({tire_life:.1f}%) - fresh tires will improve pace\n"
            if fuel_laps < laps_remaining:
                response += f"• Fuel is short ({fuel_laps:.1f} vs {laps_remaining} laps needed)\n"
            
            response += f"\n**Optimal pit window:** Next 2-3 laps\n"
            response += "**Benefits:** Fresh tires will allow you to push hard in final stint.\n"
        
        return response
    
    def answer_weather_query(self, query: str) -> str:
        """Generate answer for weather questions"""
        if not self.context:
            return "Please load race data first to check weather conditions."
        
        data = self.context['data']
        
        if data.empty:
            return "No weather data available."
        
        response = "**Weather & Track Conditions:**\n\n"
        
        if 'AIR_TEMP' in data.columns:
            air_temp = data['AIR_TEMP'].iloc[-1]
            response += f"• Air temperature: **{air_temp:.1f}°C**\n"
        
        if 'TRACK_TEMP' in data.columns:
            track_temp = data['TRACK_TEMP'].iloc[-1]
            response += f"• Track temperature: **{track_temp:.1f}°C**\n"
        
        if 'WIND_SPEED' in data.columns:
            wind = data['WIND_SPEED'].iloc[-1]
            response += f"• Wind speed: **{wind:.1f} km/h**\n"
        
        if 'HUMIDITY' in data.columns:
            humidity = data['HUMIDITY'].iloc[-1]
            response += f"• Humidity: **{humidity:.1f}%**\n"
        
        response += "\n**Impact on strategy:** "
        
        if 'TRACK_TEMP' in data.columns:
            track_temp = data['TRACK_TEMP'].iloc[-1]
            if track_temp > 35:
                response += "High track temps may accelerate tire degradation. Monitor tire life closely."
            elif track_temp < 20:
                response += "Cooler temps should help tire longevity. Tires may last longer than expected."
            else:
                response += "Temperatures are optimal for tire performance."
        
        return response
    
    def process_query(self, query: str) -> str:
        """
        Process a natural language query and return an answer
        
        Args:
            query: User's question in natural language
        
        Returns:
            Formatted answer string
        """
        # Detect intent
        intent = self.detect_intent(query)
        
        # Store in conversation history
        self.conversation_history.append({
            'query': query,
            'intent': intent
        })
        
        # Route to appropriate handler
        if intent == 'tire_status':
            answer = self.answer_tire_query(query)
        elif intent == 'pit_timing':
            answer = self.answer_pit_query(query)
        elif intent == 'fuel_status':
            answer = self.answer_fuel_query(query)
        elif intent == 'performance':
            answer = self.answer_performance_query(query)
        elif intent == 'strategy':
            answer = self.answer_strategy_query(query)
        elif intent == 'weather':
            answer = self.answer_weather_query(query)
        else:
            answer = self._get_general_answer(query)
        
        return answer
    
    def _get_general_answer(self, query: str) -> str:
        """Provide a general response for unrecognized queries"""
        return ("I can help you with:\n"
                "• **Tire status** - \"How are my tires?\"\n"
                "• **Pit timing** - \"When should I pit?\"\n"
                "• **Fuel status** - \"How much fuel do I have left?\"\n"
                "• **Performance** - \"How fast am I going?\"\n"
                "• **Strategy** - \"What strategy should I use?\"\n"
                "• **Weather** - \"What's the weather like?\"\n\n"
                "Try asking a more specific question!")
    
    def get_suggested_questions(self) -> List[str]:
        """Get list of suggested questions user can ask"""
        return [
            "How are my tires doing?",
            "When should I pit?",
            "How much fuel do I have left?",
            "What's my current pace?",
            "Can I finish without pitting?",
            "What strategy should I use?",
            "What are the weather conditions?",
            "Am I consistent?",
        ]
