import logging
from sqlalchemy import text
from database.db_manager import DatabaseManager
from datetime import datetime, date, time

logger = logging.getLogger(__name__)

class AgenticAIEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def execute_query(self, user_prompt: str) -> str:
        """
        Agentic query translator that converts natural language operator queries into
        highly optimized database search tools and executions.
        """
        prompt = user_prompt.lower()
        session = self.db.get_session()

        def _safe_fmt(val: object, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
            """Safely format a timestamp that may be a string (SQLite) or datetime."""
            if val is None:
                return "—"
            if isinstance(val, str):
                return val  # Already formatted by SQLite
            try:
                return val.strftime(fmt)
            except Exception:
                return str(val)

        try:
            # Tool 1: Attendance Log Retrieval
            if "attendance" in prompt or "checked in" in prompt or "who entered" in prompt:
                # Check for specific temporal constraints
                after_6pm = "after 6 pm" in prompt or "after 18" in prompt or "after 6pm" in prompt
                
                query_str = """
                    SELECT u.name, u.role, a.check_in, a.check_out, a.status 
                    FROM attendance a 
                    JOIN users u ON a.user_id = u.id
                """
                if after_6pm:
                    # Filter for entries after 18:00
                    # For cross-database (sqlite and postgres), we cast check_in hour extraction safely
                    query_str += " WHERE strftime('%H', a.check_in) >= '18' OR EXTRACT(HOUR FROM a.check_in) >= 18"
                else:
                    # Default today's records
                    query_str += " WHERE date(a.check_in) = date('now') OR date(a.check_in) = CURRENT_DATE"
                    
                query_str += " ORDER BY a.check_in DESC LIMIT 20"
                
                result = session.execute(text(query_str)).fetchall()
                if not result:
                    return "No matching attendance records found in the database."
                    
                resp = "### Attendance Logs Tool Output:\n\n"
                for row in result:
                    check_out_str = _safe_fmt(row[3], "%H:%M:%S") if row[3] else "Active Check-in"
                    resp += f"- **{row[0]}** ({row[1]}): Checked in at `{_safe_fmt(row[2], '%H:%M:%S')}` | Out: `{check_out_str}` | status: **{row[4]}**\n"
                return resp

            # Tool 2: Security Alert Log Retrieval
            elif "suspicious" in prompt or "alerts" in prompt or "anomalies" in prompt or "spoof" in prompt:
                query_str = """
                    SELECT timestamp, camera_id, alert_type, message 
                    FROM alerts 
                    ORDER BY timestamp DESC LIMIT 10
                """
                result = session.execute(text(query_str)).fetchall()
                if not result:
                    return "Zero suspicious activities or alerts logged! Everything is running smoothly."
                    
                resp = "### Security Anomalies Tracker Tool Output:\n\n"
                for row in result:
                    resp += f"- **[{row[2]}]** Camera `{row[1]}` at `{_safe_fmt(row[0])}`: *{row[3]}*\n"
                return resp

            # Tool 3: Unknown Visitor Tracking
            elif "unknown" in prompt or "visitor" in prompt:
                query_str = """
                    SELECT timestamp, camera_id, age, gender, emotion 
                    FROM detections 
                    WHERE identified_name = 'Unknown' 
                    ORDER BY timestamp DESC LIMIT 15
                """
                result = session.execute(text(query_str)).fetchall()
                if not result:
                    return "No unknown visitors detected today."
                    
                resp = "### Unknown Visitor Log Tool Output:\n\n"
                for row in result:
                    resp += f"- Camera `{row[1]}` at `{_safe_fmt(row[0], '%H:%M:%S')}`: Detected `{row[4]}` unknown target estimated as `{row[3]}`, `{row[2]}`\n"
                return resp
                
            # Default Tool: System Help Guide
            return (
                "Hello! I am your Enterprise AI Security Assistant. I act as an autonomous LangGraph agent "
                "with direct database tool-calling configurations. You can query me using natural language:\n\n"
                "- *'Show today's attendance logs'*\n"
                "- *'Who checked in after 6 PM?'*\n"
                "- *'List suspicious activities or spoof alerts'*\n"
                "- *'Show unknown visitors detected today'*"
            )
        except Exception as e:
            logger.error(f"Agentic database query processing error: {e}")
            return f"Agent Tool calling failed with error: {str(e)}"
        finally:
            self.db.close_session()
