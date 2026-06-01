import unittest
from database.db_manager import DatabaseManager
from backend.assistant.agent_engine import AgenticAIEngine

class TestAgenticAI(unittest.TestCase):
    def setUp(self):
        # Isolated SQLite database connection
        self.db = DatabaseManager(db_url="sqlite:///:memory:")
        self.agent = AgenticAIEngine(self.db)
        
    def test_assistant_help_fallback(self):
        # Querying an unrelated question should return the guide menu
        resp = self.agent.execute_query("what is the weather?")
        self.assertIn("Enterprise AI Security Assistant", resp)
        self.assertIn("attendance", resp)
        
    def test_attendance_queries(self):
        resp = self.agent.execute_query("who checked in today?")
        # Should execute successfully (returning zero entries initially for clean test run)
        self.assertTrue(len(resp) > 0)
        
    def test_alert_queries(self):
        resp = self.agent.execute_query("list suspicious activities")
        self.assertTrue(len(resp) > 0)

if __name__ == '__main__':
    unittest.main()
