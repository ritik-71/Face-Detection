import cv2
import time
import threading
import queue
import logging
from typing import Dict, Callable, List
import numpy as np
from database.db_manager import DatabaseManager
from ai_models.detectors.factory import get_detector
from ai_models.recognition.recognition_engine import FaceRecognitionEngine
from ai_models.liveness.liveness_v2 import LivenessDetectorV2
from ai_models.landmarks.landmarks_mesh import FaceMeshAnalytics
from ai_models.analytics.emotion_analyzer import EmotionAnalyzer
from ai_models.analytics.age_gender_estimator import AgeGenderEstimator
from ai_models.tracking.tracker import FaceTracker
from ai_models.analytics.surveillance import SurveillanceAnalytics
from backend.streams.event_publisher import EventPublisher

logger = logging.getLogger(__name__)

class CameraStreamWorker:
    def __init__(self, camera_id: str, stream_url: str, db_manager: DatabaseManager, frame_callback: Callable):
        self.camera_id = camera_id
        # Convert stream URL to integer if it is a local USB camera ID
        try:
            self.stream_url = int(stream_url)
        except ValueError:
            self.stream_url = stream_url
            
        self.db = db_manager
        self.frame_callback = frame_callback
        
        # Load AI pipeline components
        self.detector = get_detector("opencv_dnn", min_confidence=0.55)
        self.tracker = FaceTracker(max_disappeared=10)
        self.recognition_engine = FaceRecognitionEngine()
        self.liveness_detector = LivenessDetectorV2()
        self.face_mesh = FaceMeshAnalytics(max_num_faces=5)
        self.emotion_analyzer = EmotionAnalyzer()
        self.age_gender_estimator = AgeGenderEstimator()
        
        # Upgraded Surveillance & Event Hub Integrations
        self.surveillance = SurveillanceAnalytics(loitering_threshold_seconds=10.0)
        self.event_publisher = EventPublisher()
        # Default restricted zone polygon vertices in pixels
        self.restricted_zone = [(50, 50), (320, 50), (320, 320), (50, 320)]
        
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.enrolled_faces = {}
        self.last_enrolled_fetch = 0
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Ingestion stream worker started for camera: {self.camera_id}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info(f"Ingestion stream worker stopped for camera: {self.camera_id}")

    def _fetch_enrolled_embeddings(self):
        # Refresh cached embeddings every 30 seconds
        now = time.time()
        if now - self.last_enrolled_fetch > 30 or not self.enrolled_faces:
            self.enrolled_faces = self.db.get_all_enrolled_embeddings()
            self.last_enrolled_fetch = now

    def _run(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            logger.error(f"Cannot open video stream for camera {self.camera_id} at {self.stream_url}")
            self.db.log_alert(self.camera_id, "CameraDisconnected", f"Failed to connect to stream {self.stream_url}")
            self.running = False
            return
            
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 30.0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Connection lost for camera stream {self.camera_id}")
                self.db.log_alert(self.camera_id, "CameraDisconnected", "Camera returned empty frame. Stream connection lost.")
                time.sleep(2)
                # Attempt reconnection
                cap.release()
                cap = cv2.VideoCapture(self.stream_url)
                continue
                
            fps_frame_count += 1
            now = time.time()
            if now - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (now - fps_start_time)
                fps_frame_count = 0
                fps_start_time = now
                
            # Perform frame skip optimizations if pipeline queue gets full (prevents lagging)
            # Only process every 2nd or 3rd frame under heavy load, but display frames at full pace
            self._fetch_enrolled_embeddings()
            
            # 1. Face Detection
            detections = self.detector.detect(frame)
            boxes = [d['box'] for d in detections]
            
            # 2. Tracking (Persistent IDs)
            tracked_boxes = self.tracker.update(boxes)
            
            # Draw restricted zone polygon outline on incoming frame
            pts = np.array(self.restricted_zone, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(frame, "SECURE ZONE", (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            # 3. Analyze each tracked face
            active_analytics = []
            for track_id, box in tracked_boxes.items():
                x, y, w, h = box
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue
                    
                # Face Embedding & Identification
                query_emb = self.recognition_engine.get_embedding(face_crop)
                name, score = self.recognition_engine.identify_face(query_emb, self.enrolled_faces)
                
                # Dynamic Liveness V2 Checks (EAR Blink + Laplacian + Deepfake GAN Residuals)
                liveness_res = self.liveness_detector.evaluate_liveness_v2(face_crop)
                
                # Emotion, Age and Gender Predictors
                emotion_res = self.emotion_analyzer.analyze_emotion(face_crop)
                age_gender_res = self.age_gender_estimator.estimate(face_crop)
                
                # Spatial restricted zone check
                inside_zone = self.surveillance.is_inside_restricted_zone(box, self.restricted_zone)
                if inside_zone:
                    self.db.log_alert(self.camera_id, "RestrictedZoneIntrusion", f"Track ID {track_id} entered Secure Zone.")
                    self.event_publisher.publish_event("SecurityAlert", {
                        "camera_id": self.camera_id, "track_id": track_id, "alert_type": "RestrictedZoneIntrusion"
                    })
                    
                # Loitering duration checks
                loitering_alert, duration = self.surveillance.check_loitering(track_id)
                if loitering_alert:
                    self.db.log_alert(self.camera_id, "LoiteringAlert", f"Track ID {track_id} lingers: {duration}s.")
                    self.event_publisher.publish_event("SecurityAlert", {
                        "camera_id": self.camera_id, "track_id": track_id, "alert_type": "LoiteringAlert", "duration": duration
                    })
                
                # Database operations
                if name != "Unknown":
                    # Mark check-in attendance
                    attendance_res = self.db.mark_attendance(name)
                    if attendance_res.get("status") in ["check_in", "check_out"]:
                        logger.info(f"Attendance automatically recorded for {name} ({attendance_res['status']})")
                        self.event_publisher.publish_event("AttendanceMarked", {
                            "username": name, "status": attendance_res["status"]
                        })
                
                # Spoof / Deepfake trigger alerts
                if liveness_res.get("is_spoof") or liveness_res.get("deepfake_alert"):
                    alert_msg = f"Liveness/Deepfake mismatch for tracked ID: {track_id}."
                    self.db.log_alert(self.camera_id, "SpoofingAttempt", alert_msg)
                    self.event_publisher.publish_event("SecurityAlert", {
                        "camera_id": self.camera_id, "track_id": track_id, "alert_type": "SpoofingAttempt", "message": alert_msg
                    })
                    
                # Publish general detection event
                self.event_publisher.publish_event("FaceRecognized" if name != "Unknown" else "UnknownVisitor", {
                    "camera_id": self.camera_id,
                    "track_id": track_id,
                    "name": name,
                    "age": age_gender_res["age"],
                    "gender": age_gender_res["gender"],
                    "emotion": emotion_res["primary"],
                    "liveness": liveness_res["liveness_score"]
                })
                    
                # Log raw detection database entry for analytical graphs
                self.db.log_detection(
                    camera_id=self.camera_id,
                    track_id=track_id,
                    age=age_gender_res["age"],
                    gender=age_gender_res["gender"],
                    emotion=emotion_res["primary"],
                    liveness=liveness_res["liveness_score"],
                    name=name
                )
                
                # Visual enhancements: Render annotations on the display frame
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                if liveness_res.get("is_spoof") or liveness_res.get("deepfake_alert"):
                    color = (0, 165, 255) # Orange warning alert
                    
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                label_text = f"ID {track_id}: {name} ({score:.2f}) | {age_gender_res['gender']}, {age_gender_res['age']}"
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                active_analytics.append({
                    "track_id": track_id,
                    "name": name,
                    "box": box,
                    "confidence": round(score, 2),
                    "emotion": emotion_res["primary"],
                    "liveness": liveness_res["liveness_score"],
                    "is_spoof": liveness_res["is_spoof"],
                    "age": age_gender_res["age"],
                    "gender": age_gender_res["gender"],
                    "deepfake_alert": liveness_res["deepfake_alert"]
                })
                
            # Render watermark/metrics on frame
            cv2.putText(frame, f"FPS: {current_fps:.1f} | CAM: {self.camera_id}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Send the annotated frame and analytical data to the backend handler
            self.frame_callback(self.camera_id, frame, active_analytics, current_fps)
            
        cap.release()

class StreamManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.workers: Dict[str, CameraStreamWorker] = {}
        self.frame_callbacks: List[Callable] = []
        
    def add_callback(self, callback: Callable):
        self.frame_callbacks.append(callback)
        
    def _broadcast_frame(self, camera_id: str, frame: np.ndarray, analytics: List[dict], fps: float):
        for callback in self.frame_callbacks:
            try:
                callback(camera_id, frame, analytics, fps)
            except Exception as e:
                logger.error(f"Error in stream subscriber callback: {e}")
                
    def start_camera_stream(self, camera_id: str, url: str):
        if camera_id in self.workers:
            logger.info(f"Stream {camera_id} is already running. Restarting...")
            self.stop_camera_stream(camera_id)
            
        worker = CameraStreamWorker(camera_id, url, self.db, self._broadcast_frame)
        self.workers[camera_id] = worker
        worker.start()
        
    def stop_camera_stream(self, camera_id: str):
        if camera_id in self.workers:
            self.workers[camera_id].stop()
            del self.workers[camera_id]
            
    def start_all_active_cameras(self):
        cameras = self.db.get_active_cameras()
        for cam in cameras:
            logger.info(f"Launching active database camera: {cam.name} ({cam.id})")
            self.start_camera_stream(cam.id, cam.url)
            
    def stop_all_streams(self):
        for cid in list(self.workers.keys()):
            self.stop_camera_stream(cid)
