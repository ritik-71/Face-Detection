import unittest
import numpy as np
from database.db_manager import DatabaseManager

class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        # Initialize an isolated in-memory SQLite database connection for unit testing
        self.db = DatabaseManager(db_url="sqlite:///:memory:")
        
    def test_user_registration(self):
        # Register a mock face embedding
        emb = np.random.randn(128)
        user = self.db.register_user(name="John Testing", role="Employee", embedding=emb)
        self.assertIsNotNone(user.id)
        self.assertEqual(user.name, "John Testing")
        self.assertEqual(user.role, "Employee")
        
    def test_duplicate_registration_prevention(self):
        emb = np.random.randn(128)
        u1 = self.db.register_user(name="Unique User", role="Student", embedding=emb)
        u2 = self.db.register_user(name="Unique User", role="Student", embedding=emb)
        # Should return same user instance without crash
        self.assertEqual(u1.id, u2.id)

    def test_attendance_logging(self):
        emb = np.random.randn(128)
        self.db.register_user(name="Attendee A", role="Employee", embedding=emb)
        
        # Mark check-in
        res = self.db.mark_attendance("Attendee A")
        self.assertEqual(res["status"], "check_in")
        
        # Mark again today (should update checkout or return no_change depending on timestamp difference)
        res2 = self.db.mark_attendance("Attendee A")
        self.assertIn(res2["status"], ["no_change", "check_out"])

if __name__ == '__main__':
    unittest.main()
