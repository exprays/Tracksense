"""
Real-time feature engineering for live telemetry data.
Generates features for ML predictions on streaming data.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class RealtimeFeatureEngine:
    """Generate features from streaming telemetry for ML models."""
    
    def __init__(self):
        self.feature_cache = {}
        self.models = None  # Will be loaded on demand
        
    def generate_features(self, 
                         session_id: str,
                         car_id: str, 
                         current: dict,
                         history: List[dict]) -> dict:
        """
        Generate features from current message and history.
        
        Args:
            session_id: Session identifier
            car_id: Car identifier
            current: Current telemetry message
            history: Recent telemetry history (last N messages)
            
        Returns:
            Dictionary of computed features
        """
        features = {
            "session_id": session_id,
            "car_id": car_id,
            "timestamp": current.get("timestamp_utc"),
        }
        
        # Current state features
        position = current.get("position", {})
        dynamics = current.get("dynamics", {})
        
        features.update({
            # Position
            "lap": position.get("lap", 0),
            "lap_distance_m": position.get("lap_distance_m", 0),
            "lap_progress": position.get("lap_distance_m", 0) / max(position.get("track_length_m", 4200), 1),
            
            # Dynamics
            "speed_kmh": dynamics.get("speed_kmh", 0),
            "rpm": dynamics.get("rpm", 0),
            "gear": dynamics.get("gear", 0),
            "throttle_pct": dynamics.get("throttle_pedal_pct", 0),
            "brake_front_bar": dynamics.get("brake_pressure_front_bar", 0),
            "brake_rear_bar": dynamics.get("brake_pressure_rear_bar", 0),
            "accel_long_g": dynamics.get("accel_long_g", 0),
            "accel_lat_g": dynamics.get("accel_lat_g", 0),
            "steering_angle": dynamics.get("steering_angle_deg", 0),
        })
        
        # Historical features (from rolling window)
        if len(history) > 0:
            features.update(self._compute_historical_features(history))
        
        # Derived features
        features.update(self._compute_derived_features(features, history))
        
        return features
    
    def _compute_historical_features(self, history: List[dict]) -> dict:
        """Compute features from historical data."""
        speeds = [msg.get("dynamics", {}).get("speed_kmh", 0) for msg in history]
        throttles = [msg.get("dynamics", {}).get("throttle_pedal_pct", 0) for msg in history]
        brakes_f = [msg.get("dynamics", {}).get("brake_pressure_front_bar", 0) for msg in history]
        accel_longs = [msg.get("dynamics", {}).get("accel_long_g", 0) for msg in history]
        accel_lats = [msg.get("dynamics", {}).get("accel_lat_g", 0) for msg in history]
        
        return {
            # Speed statistics
            "speed_mean": np.mean(speeds) if speeds else 0,
            "speed_std": np.std(speeds) if len(speeds) > 1 else 0,
            "speed_max": np.max(speeds) if speeds else 0,
            "speed_min": np.min(speeds) if speeds else 0,
            
            # Throttle usage
            "throttle_mean": np.mean(throttles) if throttles else 0,
            "throttle_std": np.std(throttles) if len(throttles) > 1 else 0,
            
            # Braking intensity
            "brake_mean": np.mean(brakes_f) if brakes_f else 0,
            "brake_max": np.max(brakes_f) if brakes_f else 0,
            "brake_events": sum(1 for b in brakes_f if b > 5.0),
            
            # G-forces
            "accel_long_mean": np.mean(accel_longs) if accel_longs else 0,
            "accel_long_max": np.max(accel_longs) if accel_longs else 0,
            "accel_lat_mean": np.mean(np.abs(accel_lats)) if accel_lats else 0,
            "accel_lat_max": np.max(np.abs(accel_lats)) if accel_lats else 0,
            
            # Window size
            "history_window": len(history)
        }
    
    def _compute_derived_features(self, features: dict, history: List[dict]) -> dict:
        """Compute derived/engineered features."""
        derived = {}
        
        # Speed trends
        if len(history) >= 10:
            recent_speeds = [msg.get("dynamics", {}).get("speed_kmh", 0) 
                           for msg in history[-10:]]
            if len(recent_speeds) > 1:
                speed_trend = (recent_speeds[-1] - recent_speeds[0]) / len(recent_speeds)
                derived["speed_trend"] = speed_trend
                derived["is_accelerating"] = speed_trend > 0.5
                derived["is_braking_zone"] = speed_trend < -1.0
        
        # Driving aggression score
        throttle_mean = features.get("throttle_mean", 0)
        brake_max = features.get("brake_max", 0)
        accel_lat_max = features.get("accel_lat_max", 0)
        
        aggression = (throttle_mean / 100.0) * 0.4 + \
                    (min(brake_max / 50.0, 1.0)) * 0.3 + \
                    (min(accel_lat_max / 2.0, 1.0)) * 0.3
        
        derived["aggression_score"] = aggression
        
        # Tire stress indicator (simplified)
        tire_stress = (accel_lat_max * 0.5 + 
                      (brake_max / 50.0) * 0.3 + 
                      (throttle_mean / 100.0) * 0.2)
        derived["tire_stress"] = min(tire_stress, 1.0)
        
        return derived
    
    def predict(self, features: dict) -> dict:
        """
        Run ML predictions on features.
        
        Returns:
            Dictionary of predictions
        """
        predictions = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        try:
            # Lap time prediction
            predictions["predicted_lap_time"] = self._predict_lap_time(features)
            predictions["lap_time_confidence"] = 0.85
            
            # Tire degradation
            predictions["tire_deg_pct"] = self._predict_tire_deg(features)
            
            # Overtaking probability
            predictions["overtake_prob_next_3_laps"] = self._predict_overtake_prob(features)
            
            # Incident risk
            predictions["incident_risk"] = self._predict_incident_risk(features)
            predictions["incident_risk_level"] = self._get_risk_level(predictions["incident_risk"])
            
            # Strategy recommendations
            predictions["strategy_alerts"] = self._generate_strategy_alerts(features, predictions)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            predictions["error"] = str(e)
        
        return predictions
    
    def _predict_lap_time(self, features: dict) -> float:
        """Predict lap time based on current pace."""
        # Simplified prediction - replace with actual model
        speed_mean = features.get("speed_mean", 120)
        track_length = 4200  # meters
        
        # Convert km/h to m/s and estimate lap time
        speed_ms = speed_mean / 3.6
        estimated_time = track_length / speed_ms if speed_ms > 0 else 120.0
        
        # Add variation based on aggression
        aggression = features.get("aggression_score", 0.5)
        time_factor = 1.0 - (aggression - 0.5) * 0.1
        
        return estimated_time * time_factor
    
    def _predict_tire_deg(self, features: dict) -> float:
        """Predict tire degradation percentage."""
        # Simplified model based on tire stress and lap
        tire_stress = features.get("tire_stress", 0.5)
        lap = features.get("lap", 1)
        
        # Assume linear degradation with stress factor
        base_deg_per_lap = 2.0  # 2% per lap baseline
        deg = (base_deg_per_lap * lap) * (1 + tire_stress)
        
        return min(deg, 100.0)
    
    def _predict_overtake_prob(self, features: dict) -> float:
        """Predict probability of overtaking in next 3 laps."""
        # Simplified - based on pace and aggression
        speed_mean = features.get("speed_mean", 120)
        aggression = features.get("aggression_score", 0.5)
        
        # Higher speed and aggression = higher overtake probability
        base_prob = min(speed_mean / 200.0, 1.0) * 0.5
        aggression_bonus = aggression * 0.3
        
        return min(base_prob + aggression_bonus, 0.95)
    
    def _predict_incident_risk(self, features: dict) -> float:
        """Predict incident risk score (0-1)."""
        # Risk factors
        speed = features.get("speed_kmh", 0)
        brake_max = features.get("brake_max", 0)
        accel_lat_max = features.get("accel_lat_max", 0)
        tire_stress = features.get("tire_stress", 0)
        
        # High speed + hard braking + high lateral g = higher risk
        risk = 0.0
        
        if speed > 180:
            risk += 0.2
        if brake_max > 40:
            risk += 0.3
        if accel_lat_max > 1.5:
            risk += 0.3
        if tire_stress > 0.7:
            risk += 0.2
        
        return min(risk, 1.0)
    
    def _get_risk_level(self, risk: float) -> str:
        """Convert risk score to level."""
        if risk < 0.3:
            return "low"
        elif risk < 0.6:
            return "medium"
        else:
            return "high"
    
    def _generate_strategy_alerts(self, features: dict, predictions: dict) -> List[str]:
        """Generate strategic alerts based on predictions."""
        alerts = []
        
        # Tire degradation alert
        tire_deg = predictions.get("tire_deg_pct", 0)
        if tire_deg > 80:
            alerts.append("⚠️ High tire degradation - consider pit stop")
        elif tire_deg > 60:
            alerts.append("⚡ Monitor tire condition closely")
        
        # Incident risk alert
        risk_level = predictions.get("incident_risk_level", "low")
        if risk_level == "high":
            alerts.append("🚨 High incident risk detected - reduce aggression")
        
        # Overtaking opportunity
        overtake_prob = predictions.get("overtake_prob_next_3_laps", 0)
        if overtake_prob > 0.7:
            alerts.append("🎯 Good overtaking opportunity in next 3 laps")
        
        # Pace alert
        lap_time = predictions.get("predicted_lap_time", 0)
        if lap_time > 0:
            if lap_time < 90:
                alerts.append("🔥 Excellent pace maintained")
            elif lap_time > 120:
                alerts.append("📉 Pace dropping - check for issues")
        
        return alerts
