import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict

class FaceTracker:
    def __init__(self, max_disappeared: int = 15, min_iou: float = 0.3):
        """
        Initialize the tracker.
        
        Args:
            max_disappeared (int): Number of consecutive frames a face can go undetected before deletion.
            min_iou (float): Minimum Intersection-over-Union required to consider a match.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # Stores centroid of tracked IDs
        self.boxes = OrderedDict()    # Stores [x, y, w, h] of tracked IDs
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        
    def _calculate_iou(self, boxA: List[int], boxB: List[int]) -> float:
        """Compute the Intersection over Union (IoU) of two bounding boxes."""
        # box format: [x, y, w, h] -> convert to [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
            
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return float(iou)
        
    def register(self, centroid: Tuple[int, int], box: List[int]):
        """Register a new face track."""
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id: int):
        """Deregister an obsolete face track."""
        del self.objects[object_id]
        del self.boxes[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects: List[List[int]]) -> Dict[int, List[int]]:
        """
        Update the tracker state with new detections.
        
        Args:
            rects (List[List[int]]): Bounding boxes of new detections in format [x, y, w, h]
            
        Returns:
            Dict[int, List[int]]: Mapping of persistent ID -> Bounding box [x, y, w, h]
        """
        if len(rects) == 0:
            # Mark all current tracking IDs as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] >= self.max_disappeared:
                    self.deregister(object_id)
            return self.boxes

        # Compute centroids of incoming detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + (w / 2.0))
            cY = int(y + (h / 2.0))
            input_centroids[i] = (cX, cY)

        # If we are currently tracking no objects, register all incoming detections
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance matrix between old centroids and new centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Find the row and column minima
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                object_id = object_ids[row]
                
                # Check Intersection-over-Union to prevent track jumps across different faces
                iou = self._calculate_iou(self.boxes[object_id], rects[col])
                if iou < self.min_iou:
                    continue
                    
                self.objects[object_id] = input_centroids[col]
                self.boxes[object_id] = rects[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
                
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            # For tracked objects that weren't matched, increment disappearance counter
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] >= self.max_disappeared:
                    self.deregister(object_id)
                    
            # For new detections that weren't matched, register them as new tracks
            for col in unused_cols:
                self.register(input_centroids[col], rects[col])
                
        return self.boxes
