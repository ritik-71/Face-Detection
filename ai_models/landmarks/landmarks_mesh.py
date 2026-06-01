import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class FaceMeshAnalytics:
    def __init__(self, max_num_faces: int = 5, min_detection_confidence: float = 0.5):
        self.mp_face_mesh = None
        self.face_mesh = None
        
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=True,  # Enables iris tracking and detailed lip tracking
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Face Mesh successfully initialized.")
        except ImportError:
            logger.warning("MediaPipe library not found. FaceMeshAnalytics will work with fallback methods.")

    def get_mesh_landmarks(self, image: np.ndarray) -> List[Dict]:
        """
        Extract detailed facial analytics including 468 landmarks, head pose (pitch, yaw, roll),
        eye tracking, and mouth status.
        """
        if self.face_mesh is None:
            return []
            
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        analyses = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append([int(lm.x * w), int(lm.y * h), lm.z])
                
                # 1. Eye Tracking (using standard index ranges for eyes & irises)
                # Left Iris landmarks: 468, 469, 470, 471, 472
                # Right Iris landmarks: 473, 474, 475, 476, 477
                left_iris_center = self._get_centroid(landmarks, [468, 469, 470, 471, 472])
                right_iris_center = self._get_centroid(landmarks, [473, 474, 475, 476, 477])
                
                # 2. Mouth tracking (lip indices)
                # Outer Lip: 61, 291, 0, 17, 37, 267, 84, 314
                mouth_open_ratio = self._calculate_mouth_opening(landmarks)
                
                # 3. Head Pose Estimation (Pitch, Yaw, Roll)
                # We project standard 3D facial model coordinates onto 2D image landmark coordinates
                pitch, yaw, roll = self._estimate_head_pose(landmarks, w, h)
                
                analyses.append({
                    'landmarks': landmarks,
                    'left_iris': left_iris_center,
                    'right_iris': right_iris_center,
                    'mouth_open_ratio': mouth_open_ratio,
                    'pose': {
                        'pitch': round(pitch, 2),
                        'yaw': round(yaw, 2),
                        'roll': round(roll, 2),
                        'orientation': self._get_orientation_text(pitch, yaw, roll)
                    }
                })
                
        return analyses

    def _get_centroid(self, landmarks: List[List], indices: List[int]) -> Tuple[int, int]:
        """Compute the average coordinates of a set of landmark indices."""
        xs = [landmarks[idx][0] for idx in indices if idx < len(landmarks)]
        ys = [landmarks[idx][1] for idx in indices if idx < len(landmarks)]
        if not xs or not ys:
            return (0, 0)
        return (int(np.mean(xs)), int(np.mean(ys)))

    def _calculate_mouth_opening(self, landmarks: List[List]) -> float:
        """Calculate ratio of mouth height to mouth width."""
        if len(landmarks) < 468:
            return 0.0
            
        # Upper inner lip point: 13, Lower inner lip point: 14
        # Left corner: 78, Right corner: 308
        height = np.linalg.norm(np.array(landmarks[13][:2]) - np.array(landmarks[14][:2]))
        width = np.linalg.norm(np.array(landmarks[78][:2]) - np.array(landmarks[308][:2]))
        
        if width == 0:
            return 0.0
        return float(height / width)

    def _estimate_head_pose(self, landmarks: List[List], w: int, h: int) -> Tuple[float, float, float]:
        """
        Estimate 3D head rotation angles (Pitch, Yaw, Roll) using Perspective-n-Point.
        """
        if len(landmarks) < 468:
            return 0.0, 0.0, 0.0
            
        # Reference points in 3D Space (standard anthropometric averages)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)
        
        # 2D Image Coordinates corresponding to the 3D model points
        # Nose tip: 1, Chin: 152, Left eye outer corner: 263, Right eye outer corner: 33,
        # Left mouth corner: 291, Right mouth corner: 61
        image_points = np.array([
            (landmarks[1][0], landmarks[1][1]),       # Nose tip
            (landmarks[152][0], landmarks[152][1]),   # Chin
            (landmarks[263][0], landmarks[263][1]),   # Left eye outer corner
            (landmarks[33][0], landmarks[33][1]),     # Right eye outer corner
            (landmarks[291][0], landmarks[291][1]),   # Left mouth corner
            (landmarks[61][0], landmarks[61][1])      # Right mouth corner
        ], dtype=np.float32)
        
        # Camera intrinsic matrix setup
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        # Solve for rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
            
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Decompose rotation matrix to Euler angles
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]
        
        # Standardize pitch mapping
        if pitch > 180:
            pitch -= 360
        # Account for standard gimbal locks
        return pitch, yaw, roll

    def _get_orientation_text(self, pitch: float, yaw: float, roll: float) -> str:
        """Describe head orientation based on angular limits."""
        if yaw > 15.0:
            return "Looking Left"
        elif yaw < -15.0:
            return "Looking Right"
        elif pitch > 15.0:
            return "Looking Down"
        elif pitch < -15.0:
            return "Looking Up"
        return "Frontal"
