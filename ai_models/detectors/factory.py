import logging
from ai_models.detectors.base import BaseDetector
from ai_models.detectors.mediapipe_detector import MediaPipeDetector
from ai_models.detectors.opencv_dnn_detector import OpenCVDNNDetector

logger = logging.getLogger(__name__)

def get_detector(detector_type: str = "opencv_dnn", min_confidence: float = 0.5) -> BaseDetector:
    """
    Factory to retrieve configured face detector instance.
    
    Args:
        detector_type (str): Type of face detector ('mediapipe', 'opencv_dnn', 'yolov11', 'retinaface')
        min_confidence (float): Lower-bound detection confidence threshold.
        
    Returns:
        BaseDetector: Instantiated face detector.
    """
    detector_type = detector_type.lower()
    
    if detector_type == "mediapipe":
        detector = MediaPipeDetector(min_detection_confidence=min_confidence)
        if detector.face_detection is not None:
            return detector
        logger.warning("MediaPipe is unavailable, falling back to OpenCV DNN detector.")
        
    # Default/Fallback to OpenCV DNN (YuNet / Haar Cascade)
    return OpenCVDNNDetector(min_confidence=min_confidence)
