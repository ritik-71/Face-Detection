from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    role = Column(String(50), default="Employee")  # Employee, Student, Admin
    created_at = Column(DateTime, default=datetime.utcnow)
    
    embeddings = relationship("FaceEmbedding", back_populates="user", cascade="all, delete-orphan")
    attendance = relationship("Attendance", back_populates="user", cascade="all, delete-orphan")

class FaceEmbedding(Base):
    __tablename__ = 'face_embeddings'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    # Stored as serialized JSON string to support cross-compatibility between sqlite and pgvector
    embedding_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="embeddings")

class Attendance(Base):
    __tablename__ = 'attendance'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    check_in = Column(DateTime, default=datetime.utcnow)
    check_out = Column(DateTime, nullable=True)
    status = Column(String(50), default="Present")  # Present, Late, Absent
    
    user = relationship("User", back_populates="attendance")

class DetectionLog(Base):
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    camera_id = Column(String(50), nullable=False)
    track_id = Column(Integer, nullable=True)
    age = Column(String(20))
    gender = Column(String(20))
    emotion = Column(String(50))
    liveness_score = Column(Float)
    identified_name = Column(String(100), default="Unknown")

class SecurityAlert(Base):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    camera_id = Column(String(50), nullable=False)
    alert_type = Column(String(50), nullable=False)  # Spoofing, Blacklisted, CameraDisconnected
    message = Column(Text, nullable=False)
    resolved = Column(Boolean, default=False)

class CameraStream(Base):
    __tablename__ = 'cameras'
    
    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    url = Column(String(255), nullable=False)  # RTSP URL, USB Device ID (e.g. '0'), or IP stream
    is_active = Column(Boolean, default=True)
