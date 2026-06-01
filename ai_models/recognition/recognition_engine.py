import cv2
import numpy as np
import os
import logging
import urllib.request
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Pretrained MobileFaceNet ONNX model URL (lightweight and highly accurate face recognition model)
MOBILEFACENET_ONNX_URL_MAIN = "https://github.com/gcastro/mobilefacenet-onnx/raw/main/model/mobilefacenet.onnx"
MOBILEFACENET_ONNX_URL_MASTER = "https://github.com/gcastro/mobilefacenet-onnx/raw/master/model/mobilefacenet.onnx"
MOBILEFACENET_FILENAME = "mobilefacenet.onnx"

class FaceRecognitionEngine:
    def __init__(self, model_dir: str = "ai_models/weights", threshold: float = 0.6):
        """
        Initialize the Face Recognition Engine.
        
        Args:
            model_dir (str): Directory where model weights are stored.
            threshold (float): Cosine distance threshold for recognition (typically 0.5 - 0.6).
        """
        self.model_path = os.path.join(model_dir, MOBILEFACENET_FILENAME)
        self.threshold = threshold
        self.net = None
        
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            if not os.path.exists(self.model_path):
                logger.info(f"Downloading MobileFaceNet ONNX model...")
                try:
                    urllib.request.urlretrieve(MOBILEFACENET_ONNX_URL_MAIN, self.model_path)
                except Exception:
                    urllib.request.urlretrieve(MOBILEFACENET_ONNX_URL_MASTER, self.model_path)
                logger.info(f"MobileFaceNet ONNX model saved to {self.model_path}")
            
            # Load ONNX model using OpenCV DNN module (supports CUDA if OpenCV is built with CUDA)
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            
            # Try to use CUDA if available
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("CUDA backend activated for Face Recognition Engine.")
            except Exception:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("CPU backend activated for Face Recognition Engine.")
                
            logger.info("Face Recognition Engine successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load Face Recognition Engine: {e}. Fallback to mock recognition.")
            
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate a 128-dimensional embedding from a cropped BGR face image.
        """
        if self.net is None:
            # Fallback mock embedding (normalized random vector for safety/testing)
            logger.warning("Using mock embedding generator.")
            v = np.random.randn(128)
            return v / np.linalg.norm(v)
            
        try:
            # MobileFaceNet input: 112x112 normalized image (mean=127.5, scale=1/127.5)
            resized = cv2.resize(face_image, (112, 112))
            blob = cv2.dnn.blobFromImage(resized, scale=1.0/127.5, size=(112, 112), mean=(127.5, 127.5, 127.5), swapRB=True)
            
            self.net.setInput(blob)
            embedding = self.net.forward()
            
            # L2 Normalize the embedding to compute cosine similarity via dot product
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            v = np.random.randn(128)
            return v / np.linalg.norm(v)
            
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized embeddings.
        Since they are L2 normalized, similarity is simply the dot product.
        """
        return float(np.dot(emb1, emb2))
        
    def identify_face(self, query_emb: np.ndarray, enrolled_faces: Dict[str, np.ndarray]) -> Tuple[Optional[str], float]:
        """
        Identify a query face among enrolled users.
        
        Args:
            query_emb (np.ndarray): Query face embedding.
            enrolled_faces (Dict[str, np.ndarray]): Map of user_id to registered embedding.
            
        Returns:
            Tuple[Optional[str], float]: (user_id of match or None, similarity score)
        """
        if not enrolled_faces:
            return None, 0.0
            
        best_match = None
        highest_similarity = -1.0
        
        for user_id, enrolled_emb in enrolled_faces.items():
            similarity = self.compute_similarity(query_emb, enrolled_emb)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = user_id
                
        # If the highest similarity exceeds our threshold, return the match
        if highest_similarity >= self.threshold:
            return best_match, highest_similarity
            
        return "Unknown", highest_similarity
        
    def verify_faces(self, emb1: np.ndarray, emb2: np.ndarray) -> bool:
        """
        Verify if two embeddings represent the same individual.
        """
        return self.compute_similarity(emb1, emb2) >= self.threshold
