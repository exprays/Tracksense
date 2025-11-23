"""
In-memory buffer for storing real-time telemetry and predictions.
Provides fast access for live dashboard updates.
"""

import threading
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone


class TelemetryBuffer:
    """Thread-safe buffer for telemetry data and predictions."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.lock = threading.RLock()
        
        # Storage: {session_id: {car_id: deque([messages])}}
        self.telemetry: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        
        # Latest values for quick access
        self.latest_telemetry: Dict[str, Dict[str, dict]] = defaultdict(dict)
        
        # Predictions: {session_id: {car_id: latest_predictions}}
        self.predictions: Dict[str, Dict[str, dict]] = defaultdict(dict)
        
        # Session metadata
        self.sessions: Dict[str, dict] = {}
    
    def add_telemetry(self, session_id: str, car_id: str, message: dict):
        """Add telemetry message to buffer."""
        with self.lock:
            # Add to history
            history = self.telemetry[session_id][car_id]
            history.append(message)
            
            # Maintain max history size
            if len(history) > self.max_history:
                history.popleft()
            
            # Update latest
            self.latest_telemetry[session_id][car_id] = message
    
    def add_prediction(self, session_id: str, car_id: str, prediction: dict):
        """Store prediction results."""
        with self.lock:
            self.predictions[session_id][car_id] = {
                **prediction,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_latest(self, session_id: str, car_id: str) -> Optional[dict]:
        """Get latest telemetry for a car."""
        with self.lock:
            return self.latest_telemetry.get(session_id, {}).get(car_id)
    
    def get_history(self, session_id: str, car_id: str, window: int = 100) -> List[dict]:
        """Get recent telemetry history."""
        with self.lock:
            history = self.telemetry.get(session_id, {}).get(car_id, deque())
            return list(history)[-window:]
    
    def get_prediction(self, session_id: str, car_id: str) -> Optional[dict]:
        """Get latest predictions for a car."""
        with self.lock:
            return self.predictions.get(session_id, {}).get(car_id)
    
    def get_all_cars(self, session_id: str) -> List[str]:
        """Get list of all cars in a session."""
        with self.lock:
            return list(self.latest_telemetry.get(session_id, {}).keys())
    
    def get_session_stats(self, session_id: str) -> dict:
        """Get statistics for a session."""
        with self.lock:
            cars = self.get_all_cars(session_id)
            
            stats = {
                "session_id": session_id,
                "num_cars": len(cars),
                "cars": []
            }
            
            for car_id in cars:
                latest = self.get_latest(session_id, car_id)
                history_len = len(self.telemetry[session_id][car_id])
                
                stats["cars"].append({
                    "car_id": car_id,
                    "history_size": history_len,
                    "current_lap": latest.get("position", {}).get("lap") if latest else None,
                    "last_update": latest.get("timestamp_utc") if latest else None
                })
            
            return stats
    
    def clear_session(self, session_id: str):
        """Clear all data for a session."""
        with self.lock:
            if session_id in self.telemetry:
                del self.telemetry[session_id]
            if session_id in self.latest_telemetry:
                del self.latest_telemetry[session_id]
            if session_id in self.predictions:
                del self.predictions[session_id]
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all active sessions."""
        with self.lock:
            return list(self.latest_telemetry.keys())


# Global buffer instance for sharing between receiver and dashboard
_global_buffer = None

def get_global_buffer() -> TelemetryBuffer:
    """Get or create global buffer instance."""
    global _global_buffer
    if _global_buffer is None:
        _global_buffer = TelemetryBuffer()
    return _global_buffer
