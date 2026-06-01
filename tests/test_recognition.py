import unittest
import numpy as np
from ai_models.recognition.recognition_engine import FaceRecognitionEngine

class TestFaceRecognition(unittest.TestCase):
    def setUp(self):
        self.engine = FaceRecognitionEngine()
        
    def test_similarity_bounds(self):
        # Create normalized random embeddings
        v1 = np.random.randn(128)
        v1 = v1 / np.linalg.norm(v1)
        
        # Test identity similarity (should be 1.0)
        self.assertAlmostEqual(self.engine.compute_similarity(v1, v1), 1.0, places=4)
        
    def test_identification(self):
        # Enrolled face embeddings database dictionary
        v_alice = np.array([1.0] + [0.0]*127, dtype=np.float32)
        v_bob = np.array([0.0, 1.0] + [0.0]*126, dtype=np.float32)
        
        db = {
            "Alice": v_alice,
            "Bob": v_bob
        }
        
        # Query matching Alice perfectly
        match, score = self.engine.identify_face(v_alice, db)
        self.assertEqual(match, "Alice")
        self.assertGreater(score, 0.9)
        
        # Query that matches nobody
        v_unknown = np.array([0.0]*126 + [1.0, 0.0], dtype=np.float32)
        v_unknown = v_unknown / np.linalg.norm(v_unknown)
        match, score = self.engine.identify_face(v_unknown, db)
        self.assertEqual(match, "Unknown")
        self.assertLess(score, self.engine.threshold)

if __name__ == '__main__':
    unittest.main()
