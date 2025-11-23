"""
WebSocket server for receiving real-time telemetry from Go simulator.
Validates messages, stores in buffer, and triggers ML inference.
"""

import asyncio
import json
import logging
from typing import Dict, Optional
from datetime import datetime, timezone
import websockets

from .file_buffer import FileBasedBuffer
from .feature_engine import RealtimeFeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryReceiver:
    """WebSocket server that receives telemetry from simulator."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8080,
                 buffer: Optional[FileBasedBuffer] = None,
                 feature_engine: Optional[RealtimeFeatureEngine] = None):
        self.host = host
        self.port = port
        self.buffer = buffer or FileBasedBuffer()
        self.feature_engine = feature_engine or RealtimeFeatureEngine()
        
        self.active_sessions: Dict[str, dict] = {}
        self.message_count = 0
        self.error_count = 0
        
    async def handle_client(self, websocket):
        """Handle incoming WebSocket connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        
        try:
            async for message in websocket:
                await self.process_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} closed connection normally")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
            self.error_count += 1
        finally:
            logger.info(f"Client disconnected: {client_id}")
    
    async def process_message(self, raw_message: str):
        """Process incoming message from simulator."""
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type")
            
            if msg_type == "control":
                await self.handle_control_message(message)
            elif msg_type == "telemetry":
                await self.handle_telemetry_message(message)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            self.error_count += 1
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.error_count += 1
    
    async def handle_control_message(self, message: dict):
        """Handle session control messages (start/pause/stop)."""
        session_id = message.get("session_id")
        event = message.get("event")
        
        if event == "start":
            self.active_sessions[session_id] = {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "simulator": message.get("simulator", {}),
                "race": message.get("race", {}),
                "message_count": 0
            }
            logger.info(f"Session started: {session_id}")
            logger.info(f"  Track: {message.get('simulator', {}).get('track_id')}")
            logger.info(f"  Source: {message.get('simulator', {}).get('source')}")
            logger.info(f"  Speed: {message.get('simulator', {}).get('playback_speed')}x")
            
        elif event == "pause":
            logger.info(f"Session paused: {session_id}")
            
        elif event == "resume":
            logger.info(f"Session resumed: {session_id}")
            
        elif event == "stop":
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                logger.info(f"Session stopped: {session_id}")
                logger.info(f"  Messages received: {session['message_count']}")
                del self.active_sessions[session_id]
    
    async def handle_telemetry_message(self, message: dict):
        """Handle telemetry data message."""
        # Validate required fields
        required_fields = ["session_id", "car_id", "track_id", "sequence", 
                          "position", "dynamics"]
        
        if not all(field in message for field in required_fields):
            logger.error("Missing required telemetry fields")
            self.error_count += 1
            return
        
        session_id = message["session_id"]
        car_id = message["car_id"]
        
        # Update session message count
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["message_count"] += 1
        
        # Store in buffer
        self.buffer.add_telemetry(session_id, car_id, message)
        
        # Trigger feature generation and ML inference
        features = self.feature_engine.generate_features(
            session_id, 
            car_id, 
            message,
            self.buffer.get_history(session_id, car_id, window=100)
        )
        
        # Run predictions (non-blocking)
        asyncio.create_task(self.run_predictions(session_id, car_id, features))
        
        self.message_count += 1
        
        # Log progress
        if self.message_count % 100 == 0:
            logger.info(f"Processed {self.message_count} telemetry messages")
    
    async def run_predictions(self, session_id: str, car_id: str, features: dict):
        """Run ML predictions on features (async)."""
        try:
            predictions = self.feature_engine.predict(features)
            
            # Store predictions in buffer
            self.buffer.add_prediction(session_id, car_id, predictions)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
    
    async def start_server(self):
        """Start the WebSocket server."""
        async with websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=10
        ):
            logger.info(f"Telemetry receiver started on ws://{self.host}:{self.port}")
            logger.info("Waiting for simulator connections...")
            await asyncio.Future()  # Run forever
    
    def get_stats(self) -> dict:
        """Get receiver statistics."""
        return {
            "message_count": self.message_count,
            "error_count": self.error_count,
            "active_sessions": len(self.active_sessions),
            "sessions": self.active_sessions
        }


async def main():
    """Run receiver as standalone server."""
    from .file_buffer import get_file_buffer
    buffer = get_file_buffer()
    receiver = TelemetryReceiver(host="0.0.0.0", port=8080, buffer=buffer)
    await receiver.start_server()


if __name__ == "__main__":
    asyncio.run(main())
