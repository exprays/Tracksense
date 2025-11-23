"""
File-based buffer for sharing telemetry between processes.
Uses JSON files for simple cross-process data sharing with file locking.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone
import threading

# Platform-specific imports for file locking
try:
    if os.name == 'nt':
        import msvcrt
    else:
        import fcntl
except ImportError:
    pass


class FileBasedBuffer:
    """File-based buffer for cross-process telemetry sharing."""
    
    def __init__(self, data_dir: str = "live_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        
    def _get_session_file(self, session_id: str) -> Path:
        """Get file path for session data"""
        return self.data_dir / f"{session_id}.json"
    
    def _lock_file(self, file_handle):
        """Platform-independent file locking"""
        if os.name == 'nt':  # Windows
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # Unix
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
    
    def _unlock_file(self, file_handle):
        """Platform-independent file unlocking"""
        if os.name == 'nt':  # Windows
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # Unix
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
    
    def _read_json_safe(self, file_path: Path, max_retries: int = 3) -> Optional[dict]:
        """Safely read JSON with retry on corruption"""
        for attempt in range(max_retries):
            try:
                if not file_path.exists():
                    return None
                
                with open(file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait before retry
                else:
                    # File corrupted, return None
                    return None
            except Exception:
                return None
        return None
    
    def _write_json_safe(self, file_path: Path, data: dict):
        """Safely write JSON with proper file handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Write directly with exclusive access
                with open(file_path, 'w') as f:
                    json.dump(data, f)
                return
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.05)  # Wait 50ms before retry
                else:
                    raise e
    
    def add_telemetry(self, session_id: str, car_id: str, message: dict):
        """Add telemetry message to file buffer."""
        file_path = self._get_session_file(session_id)
        
        with self.lock:
            # Load existing data
            data = self._read_json_safe(file_path)
            
            if data is None:
                data = {
                    "session_id": session_id,
                    "cars": {},
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            
            # Initialize car data if needed
            if car_id not in data["cars"]:
                data["cars"][car_id] = {
                    "history": [],
                    "latest": None,
                    "prediction": None
                }
            
            # Add to history (keep last 50 to reduce file size)
            data["cars"][car_id]["history"].append(message)
            if len(data["cars"][car_id]["history"]) > 50:
                data["cars"][car_id]["history"].pop(0)
            
            # Update latest
            data["cars"][car_id]["latest"] = message
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            # Write back atomically
            self._write_json_safe(file_path, data)
    
    def add_prediction(self, session_id: str, car_id: str, prediction: dict):
        """Store prediction results."""
        file_path = self._get_session_file(session_id)
        
        with self.lock:
            data = self._read_json_safe(file_path)
            if data is None:
                return
            
            if car_id in data.get("cars", {}):
                data["cars"][car_id]["prediction"] = {
                    **prediction,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                self._write_json_safe(file_path, data)
    
    def get_latest(self, session_id: str, car_id: str) -> Optional[dict]:
        """Get latest telemetry for a car."""
        file_path = self._get_session_file(session_id)
        
        with self.lock:
            data = self._read_json_safe(file_path)
            if data is None:
                return None
            
            return data.get("cars", {}).get(car_id, {}).get("latest")
    
    def get_history(self, session_id: str, car_id: str, window: int = 50) -> List[dict]:
        """Get recent telemetry history."""
        file_path = self._get_session_file(session_id)
        
        with self.lock:
            data = self._read_json_safe(file_path)
            if data is None:
                return []
            
            history = data.get("cars", {}).get(car_id, {}).get("history", [])
            return history[-window:]
    
    def get_prediction(self, session_id: str, car_id: str) -> Optional[dict]:
        """Get latest predictions for a car."""
        file_path = self._get_session_file(session_id)
        
        with self.lock:
            data = self._read_json_safe(file_path)
            if data is None:
                return None
            
            return data.get("cars", {}).get(car_id, {}).get("prediction")
    
    def get_all_cars(self, session_id: str) -> List[str]:
        """Get list of all cars in a session."""
        file_path = self._get_session_file(session_id)
        
        with self.lock:
            data = self._read_json_safe(file_path)
            if data is None:
                return []
            
            return list(data.get("cars", {}).keys())
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all active sessions."""
        with self.lock:
            sessions = []
            for file_path in self.data_dir.glob("session_*.json"):
                sessions.append(file_path.stem)
            return sorted(sessions, reverse=True)
    
    def get_session_stats(self, session_id: str) -> dict:
        """Get statistics for a session."""
        file_path = self._get_session_file(session_id)
        
        with self.lock:
            data = self._read_json_safe(file_path)
            if data is None:
                return {"session_id": session_id, "num_cars": 0, "cars": []}
            
            cars = data.get("cars", {})
            stats = {
                "session_id": session_id,
                "num_cars": len(cars),
                "cars": []
            }
            
            for car_id, car_data in cars.items():
                latest = car_data.get("latest")
                stats["cars"].append({
                    "car_id": car_id,
                    "history_size": len(car_data.get("history", [])),
                    "current_lap": latest.get("position", {}).get("lap") if latest else None,
                    "last_update": latest.get("timestamp_utc") if latest else None
                })
            
            return stats
    
    def clear_session(self, session_id: str):
        """Clear all data for a session."""
        file_path = self._get_session_file(session_id)
        
        with self.lock:
            if file_path.exists():
                file_path.unlink()


# Global instance
_global_file_buffer = None

def get_file_buffer() -> FileBasedBuffer:
    """Get or create global file buffer instance."""
    global _global_file_buffer
    if _global_file_buffer is None:
        _global_file_buffer = FileBasedBuffer()
    return _global_file_buffer
