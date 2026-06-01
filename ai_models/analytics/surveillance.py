import time
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class SurveillanceAnalytics:
    def __init__(self, loitering_threshold_seconds: float = 10.0):
        """
        Initialize the Surveillance Analytics Engine.
        
        Args:
            loitering_threshold_seconds (float): Max allowed time (in seconds) a target can linger in the frame.
        """
        self.loitering_threshold = loitering_threshold_seconds
        # State tracking: {track_id: start_timestamp}
        self.track_start_times: Dict[int, float] = {}
        
    def check_loitering(self, track_id: int) -> Tuple[bool, float]:
        """
        Track loitering durations.
        Returns:
            Tuple[bool, float]: (is_loitering_alert, current_duration)
        """
        now = time.time()
        if track_id not in self.track_start_times:
            self.track_start_times[track_id] = now
            return False, 0.0
            
        duration = now - self.track_start_times[track_id]
        is_loitering = duration >= self.loitering_threshold
        return is_loitering, round(duration, 1)
        
    def remove_track(self, track_id: int):
        """Clean track state when target deregisters."""
        if track_id in self.track_start_times:
            del self.track_start_times[track_id]

    def is_inside_restricted_zone(self, box: List[int], zone_polygon: List[Tuple[int, int]]) -> bool:
        """
        Ray-Casting Algorithm to verify if a bounding box center sits inside a polygonal restricted zone.
        
        Args:
            box (List[int]): [x, y, w, h] face bounding box.
            zone_polygon (List[Tuple[int, int]]): A list of (x,y) vertices defining the polygon.
            
        Returns:
            bool: True if box center is inside the restricted zone.
        """
        if not zone_polygon or len(zone_polygon) < 3:
            return False
            
        # Bounding box center coordinates
        x = int(box[0] + (box[2] / 2.0))
        y = int(box[1] + (box[3] / 2.0))
        
        n = len(zone_polygon)
        inside = False
        
        p1x, p1y = zone_polygon[0]
        for i in range(n + 1):
            p2x, p2y = zone_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
