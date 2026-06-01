import os
import json
import numpy as np
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from database.models import Base, User, FaceEmbedding, Attendance, DetectionLog, SecurityAlert, CameraStream
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Default Database connection (uses local SQLite for zero-setup execution, updates dynamically to PostgreSQL via env)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///face_analytics.db")

class DatabaseManager:
    def __init__(self, db_url: str = DATABASE_URL):
        self.engine = create_engine(
            db_url, 
            connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {}
        )
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.Session = scoped_session(self.session_factory)
        
        # Initialize tables
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables initialized successfully.")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            
    def get_session(self):
        return self.Session()
        
    def close_session(self):
        self.Session.remove()

    # --- User & Embedding Operations ---
    def register_user(self, name: str, role: str, embedding: np.ndarray) -> User:
        session = self.get_session()
        try:
            # Check if user already exists to prevent duplicates
            existing = session.query(User).filter(User.name == name).first()
            if existing:
                logger.warning(f"User '{name}' is already registered. Skipping enrollment.")
                return existing
                
            user = User(name=name, role=role)
            session.add(user)
            session.commit()
            
            # Serialize the NumPy array to list string
            emb_str = json.dumps(embedding.tolist())
            face_emb = FaceEmbedding(user_id=user.id, embedding_data=emb_str)
            session.add(face_emb)
            session.commit()
            
            logger.info(f"User {name} successfully registered with embedding vector.")
            return user
        except Exception as e:
            session.rollback()
            logger.error(f"Error registering user: {e}")
            raise
        finally:
            self.close_session()

    def get_all_enrolled_embeddings(self) -> dict:
        """
        Loads all face embeddings into a dictionary map {username: embedding_numpy} for high-performance memory queries.
        """
        session = self.get_session()
        enrolled = {}
        try:
            embeddings = session.query(FaceEmbedding).all()
            for emb in embeddings:
                user = session.query(User).filter(User.id == emb.user_id).first()
                if user:
                    enrolled[user.name] = np.array(json.loads(emb.embedding_data), dtype=np.float32)
            return enrolled
        except Exception as e:
            logger.error(f"Error retrieving enrolled embeddings: {e}")
            return {}
        finally:
            self.close_session()

    # --- Attendance Module ---
    def mark_attendance(self, username: str) -> dict:
        """
        Processes check-in/check-out for user. If not checked in today, checks in.
        If already checked in, updates the check-out timestamp.
        """
        session = self.get_session()
        try:
            user = session.query(User).filter(User.name == username).first()
            if not user:
                return {"status": "error", "message": f"User {username} not found."}
                
            today = date.today()
            # Find an existing attendance record for today
            record = session.query(Attendance).filter(
                Attendance.user_id == user.id,
                Attendance.check_in >= datetime.combine(today, datetime.min.time()),
                Attendance.check_in <= datetime.combine(today, datetime.max.time())
            ).first()
            
            if not record:
                # Check-in
                record = Attendance(user_id=user.id, check_in=datetime.now())
                session.add(record)
                session.commit()
                logger.info(f"Checked in user: {username}")
                return {"status": "check_in", "time": record.check_in.strftime("%H:%M:%S")}
            else:
                # Check-out (only update if check-out was not already logged within last 1 minute to prevent noise)
                now = datetime.now()
                if record.check_out is None or (now - record.check_out).total_seconds() > 60:
                    record.check_out = now
                    session.commit()
                    logger.info(f"Checked out user: {username}")
                    return {"status": "check_out", "time": record.check_out.strftime("%H:%M:%S")}
                    
            return {"status": "no_change", "time": record.check_in.strftime("%H:%M:%S")}
        except Exception as e:
            session.rollback()
            logger.error(f"Error marking attendance: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.close_session()

    # --- Log and Alerting Operations ---
    def log_detection(self, camera_id: str, track_id: int, age: str, gender: str, emotion: str, liveness: float, name: str):
        session = self.get_session()
        try:
            log = DetectionLog(
                camera_id=camera_id,
                track_id=track_id,
                age=age,
                gender=gender,
                emotion=emotion,
                liveness_score=liveness,
                identified_name=name
            )
            session.add(log)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging detection: {e}")
        finally:
            self.close_session()

    def log_alert(self, camera_id: str, alert_type: str, message: str):
        session = self.get_session()
        try:
            alert = SecurityAlert(
                camera_id=camera_id,
                alert_type=alert_type,
                message=message
            )
            session.add(alert)
            session.commit()
            logger.warning(f"Security Alert Triggered: [{alert_type}] {message}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging security alert: {e}")
        finally:
            self.close_session()
            
    # --- Camera Ingestion Registry ---
    def register_camera(self, camera_id: str, name: str, url: str):
        session = self.get_session()
        try:
            cam = session.query(CameraStream).filter(CameraStream.id == camera_id).first()
            if not cam:
                cam = CameraStream(id=camera_id, name=name, url=url)
                session.add(cam)
            else:
                cam.name = name
                cam.url = url
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error registering camera: {e}")
        finally:
            self.close_session()
            
    def get_active_cameras(self) -> List[CameraStream]:
        session = self.get_session()
        try:
            return session.query(CameraStream).filter(CameraStream.is_active == True).all()
        finally:
            self.close_session()
