import unittest
import numpy as np
import os
import cv2
from ai_models.detectors.factory import get_detector
from ai_models.detectors.opencv_dnn_detector import OpenCVDNNDetector

class TestFaceDetectors(unittest.TestCase):
    def setUp(self):
        # Create a blank black test frame (320x320)
        self.mock_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        # Draw a white circle to represent a mock face shape
        cv2.circle(self.mock_frame, (160, 160), 60, (255, 255, 255), -1)

    def test_factory_fallback(self):
        # Requesting an invalid detector should fall back gracefully to OpenCV DNN
        detector = get_detector("invalid_name", min_confidence=0.5)
        self.assertIsNotNone(detector)
        self.assertIsInstance(detector, OpenCVDNNDetector)

    def test_detection_on_empty_frame(self):
        detector = get_detector("opencv_dnn", min_confidence=0.5)
        detections = detector.detect(self.mock_frame)
        # An empty black frame shouldn't trigger face detections, but should execute safely
        self.assertIsInstance(detections, list)

if __name__ == '__main__':
    unittest.main()
