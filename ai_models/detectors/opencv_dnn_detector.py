import cv2
import numpy as np
import os
import logging
import urllib.request
from ai_models.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

class OpenCVDNNDetector(BaseDetector):
    def __init__(self, min_confidence: float = 0.5, model_dir: str = "ai_models/weights"):
        self.min_confidence = min_confidence
        self.model_path = os.path.join(model_dir, YUNET_MODEL_FILENAME)
        self.detector = None
        self.fallback_cascade = None
        
        # Create weights directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to download/initialize YuNet DNN
        try:
            if not os.path.exists(self.model_path):
                logger.info(f"Downloading YuNet ONNX model from {YUNET_MODEL_URL}...")
                urllib.request.urlretrieve(YUNET_MODEL_URL, self.model_path)
                logger.info(f"YuNet model successfully saved to {self.model_path}")
            
            # cv2.FaceDetectorYN requires height/width during initialization
            # We initialize with a dummy size [320, 320] and resize dynamically on detection
            self.detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(320, 320),
                score_threshold=min_confidence,
                nms_threshold=0.3,
                top_k=5000
            )
            logger.info("OpenCV YuNet DNN Detector successfully initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize YuNet DNN Detector: {e}. Falling back to Haar Cascade.")
            # Load Haar Cascade as a secure fallback
            try:
                # Use current directory cascade or look for it
                cascade_path = 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.fallback_cascade = cv2.CascadeClassifier(cascade_path)
                    logger.info("Successfully loaded fallback Haar Cascade detector.")
                else:
                    logger.error("Fallback Haar Cascade model file not found.")
            except Exception as ex:
                logger.error(f"Haar Cascade fallback initialization failed: {ex}")

    def detect(self, image: np.ndarray) -> list[dict]:
        h, w, _ = image.shape
        detections = []
        
        # If YuNet DNN detector is available
        if self.detector is not None:
            try:
                # Set input size dynamically matching the frame shape
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(image)
                
                if faces is not None:
                    for face in faces:
                        # YuNet return format is a 15-element array
                        # bbox: face[0:4] = [x, y, w, h]
                        # landmarks: face[4:14] = [right_eye_x, right_eye_y, left_eye_x, ...]
                        # score: face[14]
                        x, y, width, height = map(int, face[0:4])
                        confidence = float(face[14])
                        
                        # Guard coordinates
                        x = max(0, x)
                        y = max(0, y)
                        width = min(width, w - x)
                        height = min(height, h - y)
                        
                        landmarks = []
                        # 5 primary landmarks: right eye, left eye, nose tip, right mouth corner, left mouth corner
                        for i in range(5):
                            lx = int(face[4 + i * 2])
                            ly = int(face[5 + i * 2])
                            landmarks.append([lx, ly])
                            
                        detections.append({
                            'box': [x, y, width, height],
                            'confidence': confidence,
                            'landmarks': landmarks
                        })
                return detections
            except Exception as e:
                logger.error(f"YuNet DNN detection failed: {e}. Trying fallback...")

        # Fallback to Haar Cascade
        if self.fallback_cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.fallback_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                for (x, y, width, height) in faces:
                    detections.append({
                        'box': [int(x), int(y), int(width), int(height)],
                        'confidence': 1.0,  # Haar Cascade doesn't natively return probabilistic score directly here
                        'landmarks': []
                    })
            except Exception as e:
                logger.error(f"Fallback Haar Cascade detection failed: {e}")
                
        return detections
