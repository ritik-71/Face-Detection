import json
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EventPublisher:
    def __init__(self, broker_url: str = "redis://localhost:6379"):
        """
        Initialize the Event Streaming Hub.
        Supports Redis Streams / RabbitMQ event-driven architectures.
        """
        self.broker_url = broker_url
        self.redis_client = None
        
        try:
            import redis
            # Establish dynamic connection to Redis container
            self.redis_client = redis.from_url(broker_url, socket_timeout=2)
            logger.info(f"Successfully connected to Event Streaming Hub at {broker_url}")
        except ImportError:
            logger.warning("Redis library not installed locally. EventPublisher will route events to fallback logs.")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis Event Hub: {e}. Fallback logs enabled.")

    def publish_event(self, event_type: str, data: Dict[str, Any]):
        """
        Publish an event to the Event Streaming Hub.
        
        Event Types:
            - 'FaceDetected'
            - 'FaceRecognized'
            - 'UnknownVisitor'
            - 'AttendanceMarked'
            - 'SecurityAlert'
        """
        event_payload = {
            "event_id": f"evt_{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }
        
        serialized = json.dumps(event_payload)
        
        # 1. Publish to Redis Stream if available
        if self.redis_client is not None:
            try:
                # Add to stream named 'face_events'
                self.redis_client.xadd("face_events", {"payload": serialized})
                logger.info(f"[EVENT HUB] Published event '{event_type}' to Redis Stream.")
                return
            except Exception as e:
                logger.warning(f"Failed to push to Redis Stream: {e}. Falling back...")
                
        # 2. Fallback Logger Registry
        logger.info(f"[EVENT LOGGER FALLBACK] Staged Event [{event_type}]: {serialized}")
