import unittest
import numpy as np
from ai_models.liveness.liveness_detector import LivenessDetector

class TestLivenessVerification(unittest.TestCase):
    def setUp(self):
        self.detector = LivenessDetector(ear_threshold=0.2, consecutive_frames=2)
        
    def test_ear_calculation(self):
        # Coordinates for an open eye structure
        eye_points = [
            (10, 20), (15, 15), (25, 15),
            (30, 20), (25, 25), (15, 25)
        ]
        ear = self.detector.calculate_ear(eye_points)
        self.assertGreater(ear, 0.0)
        self.assertLess(ear, 1.0)
        
    def test_blink_tracking_logic(self):
        # Open eyes coordinates (approximate EAR > 0.2)
        open_eye = [(10, 20), (15, 12), (25, 12), (30, 20), (25, 28), (15, 28)]
        # Closed eyes coordinates (approximate EAR < 0.2)
        closed_eye = [(10, 20), (15, 19), (25, 19), (30, 20), (25, 21), (15, 21)]
        
        # Frame 1: Eyes open
        blinked, _ = self.detector.check_blink(open_eye, open_eye)
        self.assertFalse(blinked)
        
        # Frame 2: Eyes closed (1st consecutive closed frame)
        blinked, _ = self.detector.check_blink(closed_eye, closed_eye)
        self.assertFalse(blinked)
        
        # Frame 3: Eyes closed (2nd consecutive closed frame)
        blinked, _ = self.detector.check_blink(closed_eye, closed_eye)
        self.assertFalse(blinked)
        
        # Frame 4: Eyes open (triggers blink release registration)
        blinked, _ = self.detector.check_blink(open_eye, open_eye)
        self.assertTrue(blinked)
        self.assertEqual(self.detector.total_blinks, 1)

    def test_texture_spoof_detection(self):
        # Create a plain solid frame represent spoof print blur
        spoof_img = np.zeros((100, 100, 3), dtype=np.uint8)
        score = self.detector.evaluate_texture_liveness(spoof_img)
        # Should be classified as low liveness score (printed/blurred)
        self.assertLess(score, 0.5)

if __name__ == '__main__':
    unittest.main()
