import unittest
from ai_models.tracking.tracker import FaceTracker

class TestFaceTracking(unittest.TestCase):
    def setUp(self):
        self.tracker = FaceTracker(max_disappeared=3, min_iou=0.3)
        
    def test_registration(self):
        # Register a bounding box
        rects = [[10, 10, 50, 50]]
        tracks = self.tracker.update(rects)
        self.assertEqual(len(tracks), 1)
        self.assertIn(0, tracks)  # First persistent ID should be 0
        self.assertEqual(tracks[0], [10, 10, 50, 50])
        
    def test_tracking_association(self):
        # Frame 1: Face detected at starting position
        self.tracker.update([[10, 10, 50, 50]])
        
        # Frame 2: Face moved slightly (high overlap IoU)
        tracks = self.tracker.update([[12, 12, 50, 50]])
        self.assertEqual(len(tracks), 1)
        self.assertIn(0, tracks)  # ID 0 should remain active
        self.assertEqual(tracks[0], [12, 12, 50, 50])

    def test_disappearance(self):
        # Frame 1: Face detected
        self.tracker.update([[10, 10, 50, 50]])
        
        # Frame 2-3: Face disappears (holding track state)
        self.tracker.update([])
        self.tracker.update([])
        self.assertIn(0, self.tracker.objects)
        
        # Frame 4: Disappears for 3rd frame (exceeds max_disappeared)
        self.tracker.update([])
        self.assertNotIn(0, self.tracker.objects)

if __name__ == '__main__':
    unittest.main()
