import cv2
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)

class LivenessDetector:
    def __init__(self, ear_threshold: float = 0.2, consecutive_frames: int = 3):
        """
        Initialize the Anti-Spoofing & Liveness Detector.
        
        Args:
            ear_threshold (float): Eye Aspect Ratio limit below which eye is considered closed.
            consecutive_frames (int): Number of consecutive frames needed to count as a blink.
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        
        # State tracking for dynamic blink verification
        self.blink_counter = 0
        self.total_blinks = 0
        self.liveness_history = []
        
    def calculate_ear(self, eye_points: List[Tuple[int, int]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) using 6 landmark coordinates:
        Points 0 and 3 are horizontal corners, 1 & 5 and 2 & 4 are vertical pairs.
        """
        if len(eye_points) < 6:
            return 1.0
            
        p = [np.array(pt) for pt in eye_points]
        
        # Vertical distances
        a = np.linalg.norm(p[1] - p[5])
        b = np.linalg.norm(p[2] - p[4])
        
        # Horizontal distance
        c = np.linalg.norm(p[0] - p[3])
        
        # Eye aspect ratio formula
        ear = (a + b) / (2.0 * c)
        return float(ear)
        
    def check_blink(self, left_eye: List[Tuple[int, int]], right_eye: List[Tuple[int, int]]) -> Tuple[bool, float]:
        """
        Processes eye aspect ratio to detect blinks.
        Returns:
            Tuple[bool, float]: (is_blink_event, average_ear)
        """
        ear_left = self.calculate_ear(left_eye)
        ear_right = self.calculate_ear(right_eye)
        avg_ear = (ear_left + ear_right) / 2.0
        
        blink_detected = False
        
        if avg_ear < self.ear_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.consecutive_frames:
                self.total_blinks += 1
                blink_detected = True
            self.blink_counter = 0
            
        return blink_detected, avg_ear
        
    def evaluate_texture_liveness(self, face_crop: np.ndarray) -> float:
        """
        Evaluate liveness using local binary patterns or image frequency analysis.
        Spoof attacks (photos, screen displays) lack high-frequency color variations
        and display screen refresh grids or reflection glares.
        
        Returns:
            float: Liveness score (0.0 = spoof, 1.0 = alive)
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0
            
        try:
            # Convert crop to grayscale and compute Laplacian variance (focus/blur metrics)
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Analyze frequency spectrum (spoons and photos have flat frequency profiles or specific noise patterns)
            # Normalizing variance between 0.0 and 1.0
            # Photos tend to be blurry or heavily smoothed, giving lower Laplacian variance
            # Screens have high variance (moire patterns) or low variance (blur)
            # Thresholding variance to yield a liveness score
            # Real faces usually sit in a medium-high variance sweet spot (e.g., 80 - 450)
            score = 1.0
            if laplacian_var < 45.0:
                score = 0.2  # Blurry image / printed photo attack
            elif laplacian_var > 650.0:
                score = 0.3  # Screen pixel grid / moire interference patterns
            else:
                # Map standard values to a liveness probability
                score = min(1.0, 0.4 + (laplacian_var / 800.0))
                
            return float(score)
        except Exception:
            return 0.5
            
    def analyze_liveness(self, face_crop: np.ndarray, left_eye: List[Tuple[int, int]] = None, right_eye: List[Tuple[int, int]] = None) -> dict:
        """
        Comprehensive liveness analysis interface.
        
        Returns:
            dict: Liveness results including scores, blink counts, and spoof flag.
        """
        texture_score = self.evaluate_texture_liveness(face_crop)
        
        blink_event = False
        avg_ear = 1.0
        if left_eye and right_eye:
            blink_event, avg_ear = self.check_blink(left_eye, right_eye)
            
        # Liveness decision formula
        liveness_score = texture_score
        # If user is blinking, highly positive signal for life
        if blink_event:
            liveness_score = min(1.0, liveness_score + 0.3)
            
        is_spoof = liveness_score < 0.5
        
        return {
            'liveness_score': round(liveness_score, 3),
            'avg_ear': round(avg_ear, 3),
            'blink_detected': blink_event,
            'total_blinks': self.total_blinks,
            'is_spoof': is_spoof
        }
