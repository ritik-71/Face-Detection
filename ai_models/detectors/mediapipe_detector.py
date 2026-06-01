import cv2
import numpy as np
import logging
from ai_models.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

class MediaPipeDetector(BaseDetector):
    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self.mp_face_detection = None
        self.face_detection = None
        
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            # Model selection: 0 is short-range (within 2 meters), 1 is full-range (within 5 meters)
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_detection_confidence
            )
            logger.info("MediaPipe Face Detection successfully initialized.")
        except ImportError:
            logger.warning("MediaPipe library not found. MediaPipeDetector will not function unless installed.")
            
    def detect(self, image: np.ndarray) -> list[dict]:
        if self.face_detection is None:
            logger.error("MediaPipe is not installed or failed to initialize.")
            return []
            
        h, w, _ = image.shape
        # MediaPipe requires RGB images
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        detections = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Keep box within image dimensions
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                confidence = detection.score[0] if detection.score else 0.0
                
                # Extract landmarks if available
                landmarks = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        landmarks.append([int(keypoint.x * w), int(keypoint.y * h)])
                
                detections.append({
                    'box': [x, y, width, height],
                    'confidence': float(confidence),
                    'landmarks': landmarks
                })
                
        return detections
