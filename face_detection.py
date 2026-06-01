#!/usr/bin/env python3
"""
Upgraded Desktop Entry Point - AI Face Analytics Platform (2026)
Runs a comprehensive, real-time face analytics pipeline in a local GUI window,
incorporating detection, tracking, mesh landmarks, age/gender, emotion, and liveness scoring.
"""

import cv2
import time
import logging
from database.db_manager import DatabaseManager
from ai_models.detectors.factory import get_detector
from ai_models.recognition.recognition_engine import FaceRecognitionEngine
from ai_models.liveness.liveness_detector import LivenessDetector
from ai_models.landmarks.landmarks_mesh import FaceMeshAnalytics
from ai_models.analytics.emotion_analyzer import EmotionAnalyzer
from ai_models.analytics.age_gender_estimator import AgeGenderEstimator
from ai_models.tracking.tracker import FaceTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("desktop_launcher")

def main():
    logger.info("Initializing Desktop AI Face Analytics Pipeline...")
    
    # 1. Initialize local DB and components
    db = DatabaseManager()
    detector = get_detector("opencv_dnn", min_confidence=0.55)
    tracker = FaceTracker(max_disappeared=12)
    recognition_engine = FaceRecognitionEngine()
    liveness_detector = LivenessDetector()
    face_mesh_engine = FaceMeshAnalytics(max_num_faces=5)
    emotion_analyzer = EmotionAnalyzer()
    age_gender_estimator = AgeGenderEstimator()
    
    # Cache registered face embeddings from database
    enrolled_faces = db.get_all_enrolled_embeddings()
    logger.info(f"Loaded {len(enrolled_faces)} registered embeddings from database.")
    
    # 2. Start webcam video feed (0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam device '0' could not be opened. Check hardware connections.")
        return
        
    logger.info("Webcam active. Press 'ESC' key in the video window to stop.")
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture video frame.")
            break
            
        h, w, _ = frame.shape
        
        # 3. Running Core Face Detection
        detections = detector.detect(frame)
        boxes = [d['box'] for d in detections]
        
        # 4. Running Multi-Target Persistent Tracking
        tracked_boxes = tracker.update(boxes)
        
        # 5. Extract Detailed Face Mesh (468 landmarks)
        mesh_results = face_mesh_engine.get_mesh_landmarks(frame)
        
        # Draw 468 landmark mesh overlay (if landmarks detected)
        for mesh in mesh_results:
            for pt in mesh['landmarks']:
                cv2.circle(frame, (pt[0], pt[1]), 1, (0, 242, 254), -1)
                
        # 6. Analyze Each Tracked Individual
        for track_id, box in tracked_boxes.items():
            x, y, width, height = box
            face_crop = frame[y:y+height, x:x+width]
            
            if face_crop.size == 0:
                continue
                
            # Face recognition embedding identification
            query_emb = recognition_engine.get_embedding(face_crop)
            name, similarity = recognition_engine.identify_face(query_emb, enrolled_faces)
            
            # Anti-spoofing liveness checks
            liveness_res = liveness_detector.analyze_liveness(face_crop)
            
            # Emotion & demographic profiling
            emotion_res = emotion_analyzer.analyze_emotion(face_crop)
            demographics = age_gender_estimator.estimate(face_crop)
            
            # Decide color based on identity & safety state
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green vs Red
            if liveness_res.get("is_spoof"):
                color = (0, 165, 255) # Orange warning for spoofing
                
            # Draw Face Box Bounding Rectangles
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            
            # Display stats labels directly above face
            name_label = f"ID {track_id}: {name} ({similarity:.2f})"
            stats_label = f"{demographics['gender']}, {demographics['age']} | {emotion_res['primary']}"
            liveness_label = f"Liveness: {liveness_res['liveness_score']} | {'SPOOF' if liveness_res['is_spoof'] else 'LIVE'}"
            
            cv2.putText(frame, name_label, (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(frame, stats_label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(frame, liveness_label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
        # Compute and render active system FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {fps:.1f} | ACTIVE TRACKS: {len(tracked_boxes)}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display GUI
        cv2.imshow('Enterprise AI Face Analytics - Desktop Console', frame)
        
        # Press 'ESC' key to terminate the local run loop
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Desktop console terminated. Cleaned registers.")

if __name__ == '__main__':
    main()