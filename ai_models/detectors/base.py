from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detect faces in a given BGR image frame.
        
        Args:
            image (np.ndarray): Input frame in BGR format.
            
        Returns:
            list[dict]: A list of detected faces where each dictionary contains:
                - 'box': list of [x, y, w, h] (bounding box coordinates in pixels)
                - 'confidence': float (detection confidence score between 0.0 and 1.0)
                - 'landmarks': optional list of dicts or list of [x, y] coordinates
        """
        pass
