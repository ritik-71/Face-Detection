import os
import cv2
import json
import base64
import asyncio
import logging
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, date, timedelta
import numpy as np

from database.db_manager import DatabaseManager
from backend.streams.stream_manager import StreamManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("face_analytics_backend")

app = FastAPI(
    title="Enterprise AI Face Analytics Platform",
    description="Production-grade AI Video Ingestion, Face Recognition, Liveness, and Security Alert API",
    version="2026.1.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database and Multi-Camera Managers
db = DatabaseManager()
stream_manager = StreamManager(db)

# Pre-populate with default webcam if database is empty
try:
    session = db.get_session()
    from database.models import CameraStream
    if session.query(CameraStream).count() == 0:
        db.register_camera("cam_01", "Default Webcam", "0")
        logger.info("Registered default system webcam 'cam_01' into database camera streams.")
except Exception as e:
    logger.error(f"Error checking camera seeds: {e}")

# Start active cameras on launch
@app.on_event("startup")
def startup_event():
    logger.info("FastAPI service starting up...")
    stream_manager.start_all_active_cameras()

@app.on_event("shutdown")
def shutdown_event():
    logger.info("FastAPI service shutting down...")
    stream_manager.stop_all_streams()

# --- Pydantic Schemas ---
class CameraRegister(BaseModel):
    id: str
    name: str
    url: str

class UserEnrollRequest(BaseModel):
    name: str
    role: str
    image_base64: str  # Cropped or full face capture image from enrollment client

class AIQueryRequest(BaseModel):
    query: str

# --- Dynamic Frame Subscriptions for WebSockets ---
# Holds list of active WebSocket connections subscribing to each camera
camera_subscribers: Dict[str, List[WebSocket]] = {}
camera_subscribers_lock = threading.Lock()

def ws_frame_dispatcher(camera_id: str, frame: np.ndarray, analytics: List[dict], fps: float):
    """
    Subscribed callback to the StreamManager frame output.
    Encodes processed frame to JPEG base64 and distributes to connected WebSockets.
    """
    with camera_subscribers_lock:
        subscribers = camera_subscribers.get(camera_id, [])
        if not subscribers:
            return
            
    # Encode frame to JPEG
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return
        
    base64_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    payload = json.dumps({
        "camera_id": camera_id,
        "frame": f"data:image/jpeg;base64,{base64_frame}",
        "analytics": analytics,
        "fps": round(fps, 1)
    })
    
    # Broadcast to all websocket connections asynchronously
    loop = asyncio.get_event_loop()
    for ws in list(subscribers):
        asyncio.run_coroutine_threadsafe(send_ws_message(ws, payload, camera_id), loop)

async def send_ws_message(ws: WebSocket, payload: str, camera_id: str):
    try:
        await ws.send_text(payload)
    except Exception:
        # Client disconnected
        with camera_subscribers_lock:
            if camera_id in camera_subscribers and ws in camera_subscribers[camera_id]:
                camera_subscribers[camera_id].remove(ws)

# Register the WebSocket dispatcher callback with the StreamManager
stream_manager.add_callback(ws_frame_dispatcher)


# ====================================================
# REST API ENDPOINTS
# ====================================================

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# --- Camera Management API ---
@app.get("/api/cameras")
def get_cameras():
    session = db.get_session()
    try:
        from database.models import CameraStream
        cams = session.query(CameraStream).all()
        return [{"id": c.id, "name": c.name, "url": c.url, "is_active": c.is_active} for c in cams]
    finally:
        db.close_session()

@app.post("/api/cameras")
def register_camera(cam: CameraRegister):
    db.register_camera(cam.id, cam.name, cam.url)
    stream_manager.start_camera_stream(cam.id, cam.url)
    return {"status": "success", "message": f"Camera '{cam.name}' successfully added and streaming."}

@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: str):
    session = db.get_session()
    try:
        from database.models import CameraStream
        cam = session.query(CameraStream).filter(CameraStream.id == camera_id).first()
        if not cam:
            raise HTTPException(status_code=404, detail="Camera not found")
        stream_manager.stop_camera_stream(camera_id)
        session.delete(cam)
        session.commit()
        return {"status": "success", "message": f"Camera {camera_id} deleted."}
    finally:
        db.close_session()

# --- User & Embedding Enrollment API ---
@app.post("/api/users/enroll")
def enroll_user(user_req: UserEnrollRequest):
    try:
        # Decode base64 image data
        img_data = base64.b64decode(user_req.image_base64.split(",")[-1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.imread)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file encoded in base64.")
            
        # Run detection to extract crop
        from ai_models.detectors.factory import get_detector
        detector = get_detector("opencv_dnn")
        detections = detector.detect(img)
        
        if not detections:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded registration image.")
            
        # Crop the largest face detected
        largest_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        x, y, w, h = largest_face['box']
        face_crop = img[y:y+h, x:x+w]
        
        # Generate embedding vector using our RecognitionEngine
        from ai_models.recognition.recognition_engine import FaceRecognitionEngine
        engine = FaceRecognitionEngine()
        embedding = engine.get_embedding(face_crop)
        
        # Save user and embedding into PostgreSQL/SQLite
        user = db.register_user(name=user_req.name, role=user_req.role, embedding=embedding)
        
        return {
            "status": "success", 
            "message": f"Successfully enrolled face for {user_req.name} with role {user_req.role}.",
            "user_id": user.id
        }
    except Exception as e:
        logger.error(f"Enrollment failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
def list_users():
    session = db.get_session()
    try:
        from database.models import User
        users = session.query(User).all()
        return [{"id": u.id, "name": u.name, "role": u.role, "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S")} for u in users]
    finally:
        db.close_session()

# --- Attendance Management API ---
@app.get("/api/attendance")
def get_attendance(date_str: Optional[str] = Query(None)):
    session = db.get_session()
    try:
        from database.models import Attendance, User
        query = session.query(Attendance, User).join(User, Attendance.user_id == User.id)
        
        if date_str:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            query = query.filter(
                Attendance.check_in >= datetime.combine(target_date, datetime.min.time()),
                Attendance.check_in <= datetime.combine(target_date, datetime.max.time())
            )
            
        records = query.order_by(Attendance.check_in.desc()).all()
        
        output = []
        for att, user in records:
            output.append({
                "id": att.id,
                "user_id": user.id,
                "name": user.name,
                "role": user.role,
                "check_in": att.check_in.strftime("%Y-%m-%d %H:%M:%S"),
                "check_out": att.check_out.strftime("%Y-%m-%d %H:%M:%S") if att.check_out else "Active Status",
                "status": att.status
            })
        return output
    finally:
        db.close_session()

# --- Analytics Dashboard API ---
@app.get("/api/analytics/dashboard")
def get_dashboard_metrics():
    session = db.get_session()
    try:
        from database.models import User, Attendance, DetectionLog, SecurityAlert
        
        today = date.today()
        today_start = datetime.combine(today, datetime.min.time())
        
        total_registered = session.query(User).count()
        today_attendance = session.query(Attendance).filter(Attendance.check_in >= today_start).count()
        today_alerts = session.query(SecurityAlert).filter(SecurityAlert.timestamp >= today_start).count()
        total_detections_today = session.query(DetectionLog).filter(DetectionLog.timestamp >= today_start).count()
        
        # Gender demographics distribution
        genders = {"Male": 0, "Female": 0, "Unknown": 0}
        gender_logs = session.query(DetectionLog.gender).filter(DetectionLog.timestamp >= today_start).all()
        for log in gender_logs:
            if log[0] in genders:
                genders[log[0]] += 1
                
        # Emotions distribution
        emotions = {}
        emotion_logs = session.query(DetectionLog.emotion).filter(DetectionLog.timestamp >= today_start).all()
        for log in emotion_logs:
            emotions[log[0]] = emotions.get(log[0], 0) + 1
            
        # Active camera count
        from database.models import CameraStream
        active_cams = session.query(CameraStream).filter(CameraStream.is_active == True).count()
        
        # System Resource Usage (CPU, RAM, GPU fallback)
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        return {
            "total_registered": total_registered,
            "today_attendance": today_attendance,
            "active_cameras": active_cams,
            "today_alerts": today_alerts,
            "total_detections_today": total_detections_today,
            "demographics": genders,
            "emotions": emotions,
            "system_resources": {
                "cpu": cpu_usage,
                "ram": ram_usage,
                "gpu": 0.0  # Placeholder representing direct GPU execution metric
            }
        }
    finally:
        db.close_session()

# --- Security Alerts API ---
@app.get("/api/alerts")
def get_alerts():
    session = db.get_session()
    try:
        from database.models import SecurityAlert
        alerts = session.query(SecurityAlert).order_by(SecurityAlert.timestamp.desc()).limit(50).all()
        return [{
            "id": a.id,
            "timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": a.camera_id,
            "type": a.alert_type,
            "message": a.message,
            "resolved": a.resolved
        } for a in alerts]
    finally:
        db.close_session()

# --- Generative AI NLP Query Router Assistant API ---
@app.post("/api/assistant/query")
def query_assistant(req: AIQueryRequest):
    """
    Intelligent generative NLP query processing engine parsing language queries into active SQL reporting counts.
    Supports queries like "Who checked in today?", "How many security alerts?", "Show attendance", "Show active cameras".
    """
    session = db.get_session()
    try:
        q = req.query.lower()
        from database.models import Attendance, User, SecurityAlert, CameraStream
        
        if "attendance" in q or "checked in" in q or "who entered" in q:
            records = session.query(Attendance, User).join(User, Attendance.user_id == User.id).order_by(Attendance.check_in.desc()).limit(10).all()
            if not records:
                return {"response": "Nobody has checked in or out today yet."}
                
            resp = "Here is the recent attendance log:\n\n"
            for att, u in records:
                status = f"Checked out at {att.check_out.strftime('%H:%M:%S')}" if att.check_out else "Active Check-in"
                resp += f"- **{u.name}** ({u.role}): Entered at {att.check_in.strftime('%H:%M:%S')} | status: {status}\n"
            return {"response": resp}
            
        elif "alert" in q or "security" in q or "spoof" in q:
            alerts = session.query(SecurityAlert).order_by(SecurityAlert.timestamp.desc()).limit(5).all()
            if not alerts:
                return {"response": "Zero security anomalies logged! Everything looks safe."}
            resp = "Security alerts logged today:\n\n"
            for a in alerts:
                resp += f"- **[{a.alert_type}]** {a.message} (Triggered on Camera: {a.camera_id} at {a.timestamp.strftime('%H:%M:%S')})\n"
            return {"response": resp}
            
        elif "camera" in q or "stream" in q:
            cams = session.query(CameraStream).all()
            resp = "Registered video feeds:\n\n"
            for c in cams:
                status = "Online & Running" if c.is_active else "Offline"
                resp += f"- **{c.name}** (ID: {c.id}): URL `{c.url}` | Status: **{status}**\n"
            return {"response": resp}
            
        return {
            "response": "Hello! I am your AI Face Platform Assistant. I support database queries like:\n"
                        "- 'Show today's attendance logs'\n"
                        "- 'Are there any security alerts?'\n"
                        "- 'Which cameras are currently online?'"
        }
    finally:
        db.close_session()


# ====================================================
# WEBSOCKET STREAMING GATEWAY
# ====================================================

@app.websocket("/api/ws/monitor/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    
    with camera_subscribers_lock:
        if camera_id not in camera_subscribers:
            camera_subscribers[camera_id] = []
        camera_subscribers[camera_id].append(websocket)
        logger.info(f"New client subscribed to WebSockets channel for camera: {camera_id}")
        
    try:
        # Keep connection open and alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        with camera_subscribers_lock:
            if camera_id in camera_subscribers and websocket in camera_subscribers[camera_id]:
                camera_subscribers[camera_id].remove(websocket)
        logger.info(f"Client unsubscribed from WebSockets channel for camera: {camera_id}")
