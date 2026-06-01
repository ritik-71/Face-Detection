import unittest
import numpy as np
from ai_models.liveness.liveness_v2 import LivenessDetectorV2

class TestLivenessV2(unittest.TestCase):
    def setUp(self):
        self.detector = LivenessDetectorV2(ear_threshold=0.2, consecutive_frames=2)
        
    def test_deepfake_residuals(self):
        # Create solid mock frame (representing a highly GAN-smoothed blank face)
        mock_gan_face = np.zeros((128, 128, 3), dtype=np.uint8)
        res = self.detector.detect_deepfake_or_swap(mock_gan_face)
        # Blank/uniform frames lack high-frequency detail -> deepfake signature
        self.assertTrue(res["is_deepfake"])
        self.assertGreater(res["confidence"], 0.5)

    def test_liveness_v2_compilation(self):
        mock_face = np.zeros((100, 100, 3), dtype=np.uint8)
        res = self.detector.evaluate_liveness_v2(mock_face)
        self.assertIn("deepfake_alert", res)
        self.assertIn("liveness_score", res)

if __name__ == '__main__':
    unittest.main()
