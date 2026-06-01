import unittest
import time
from ai_models.analytics.surveillance import SurveillanceAnalytics

class TestSurveillanceAnalytics(unittest.TestCase):
    def setUp(self):
        self.surveillance = SurveillanceAnalytics(loitering_threshold_seconds=1.0)
        
    def test_loitering_timer(self):
        # First track detection
        alert, duration = self.surveillance.check_loitering(99)
        self.assertFalse(alert)
        self.assertEqual(duration, 0.0)
        
        # Fast delay
        time.sleep(1.1)
        alert2, duration2 = self.surveillance.check_loitering(99)
        # Should now exceed 1.0 second threshold
        self.assertTrue(alert2)
        self.assertGreaterEqual(duration2, 1.0)
        
    def test_ray_casting_zone(self):
        # A 100x100 secure box zone at x:10-110, y:10-110
        polygon = [(10, 10), (110, 10), (110, 110), (10, 110)]
        
        # Center of this box [50, 50, 20, 20] -> center is 60,60 which is INSIDE
        box_inside = [50, 50, 20, 20]
        self.assertTrue(self.surveillance.is_inside_restricted_zone(box_inside, polygon))
        
        # Center of this box [200, 200, 20, 20] -> center is 210,210 which is OUTSIDE
        box_outside = [200, 200, 20, 20]
        self.assertFalse(self.surveillance.is_inside_restricted_zone(box_outside, polygon))

if __name__ == '__main__':
    unittest.main()
