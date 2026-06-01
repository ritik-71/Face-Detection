import base64
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from database.db_manager import DatabaseManager
from ai_models.recognition.recognition_engine import FaceRecognitionEngine

logger = logging.getLogger(__name__)

class MultimodalFaceSearch:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.engine = FaceRecognitionEngine()
        
    def search_by_image_base64(self, image_base64: str) -> dict:
        """
        Upload image -> Find matching registered person using vector similarity indices.
        
        Args:
            image_base64 (str): Base64 encoded snap of the face to search for.
            
        Returns:
            dict: Search results indicating match status, user names, roles, and confidence scores.
        """
        try:
            # Decode the base64 query image
            img_data = base64.b64decode(image_base64.split(",")[-1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"status": "error", "message": "Failed to decode input search image."}
                
            # Perform a quick face crop using the detector factory
            from ai_models.detectors.factory import get_detector
            detector = get_detector("opencv_dnn", min_confidence=0.5)
            detections = detector.detect(img)
            
            if not detections:
                return {"status": "error", "message": "No face detected in the query image."}
                
            # Focus on the largest face detected
            largest_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
            x, y, w, h = largest_face['box']
            face_crop = img[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                return {"status": "error", "message": "Face crop contains zero pixels."}
                
            # Generate the 128-dimensional Normalized MobileFaceNet embedding vector
            query_emb = self.engine.get_embedding(face_crop)
            
            # Fetch all enrolled face profiles from the database
            enrolled = self.db.get_all_enrolled_embeddings()
            
            if not enrolled:
                return {"status": "no_enrolled", "message": "Database enrollment profiles are empty."}
                
            # Execute vectorized cosine similarity queries across the database embeddings
            match_name, score = self.engine.identify_face(query_emb, enrolled)
            
            if match_name == "Unknown":
                return {
                    "status": "unidentified",
                    "message": "Face search completed: Target face does not match any registered profiles.",
                    "best_similarity": round(score, 3)
                }
                
            # Retrieve detailed profile role from user table
            session = self.db.get_session()
            from database.models import User
            user = session.query(User).filter(User.name == match_name).first()
            role = user.role if user else "Employee"
            
            return {
                "status": "success",
                "match_name": match_name,
                "role": role,
                "similarity_score": round(score, 3),
                "message": f"Successfully identified target: matched '{match_name}' ({role}) with {score:.2f} confidence."
            }
        except Exception as e:
            logger.error(f"Multimodal image search error: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.db.close_session()
