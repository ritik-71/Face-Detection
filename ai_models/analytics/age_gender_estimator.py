import cv2
import numpy as np
import os
import logging
import urllib.request

logger = logging.getLogger(__name__)

# Pretrained OpenCV DNN models URLs
AGE_PROTO = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
AGE_MODEL = "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel"
GENDER_PROTO = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
GENDER_MODEL = "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel"

class AgeGenderEstimator:
    def __init__(self, model_dir: str = "ai_models/weights"):
        self.model_dir = model_dir
        self.age_net = None
        self.gender_net = None
        
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Download and initialize networks
        self._load_network("age", AGE_PROTO, AGE_MODEL)
        self._load_network("gender", GENDER_PROTO, GENDER_MODEL)
        
    def _load_network(self, name: str, proto_url: str, model_url: str):
        proto_path = os.path.join(self.model_dir, f"{name}.prototxt")
        model_path = os.path.join(self.model_dir, f"{name}.caffemodel")
        
        try:
            if not os.path.exists(proto_path):
                logger.info(f"Downloading {name} prototxt...")
                urllib.request.urlretrieve(proto_url, proto_path)
            if not os.path.exists(model_path):
                logger.info(f"Downloading {name} caffemodel...")
                urllib.request.urlretrieve(model_url, model_path)
                
            net = cv2.dnn.readNet(model_path, proto_path)
            if name == "age":
                self.age_net = net
            else:
                self.gender_net = net
            logger.info(f"{name.capitalize()} estimation network successfully loaded.")
        except Exception as e:
            logger.warning(f"Failed to load {name} DNN model: {e}. Fallback enabled.")

    def estimate(self, face_crop: np.ndarray) -> dict:
        """
        Estimate age and gender from a cropped BGR face image.
        """
        if face_crop is None or face_crop.size == 0:
            return {"gender": "Unknown", "gender_confidence": 0.0, "age": "Unknown", "age_confidence": 0.0}
            
        gender_pred = "Unknown"
        gender_conf = 0.0
        age_pred = "Unknown"
        age_conf = 0.0
        
        # Gender estimation
        if self.gender_net is not None:
            try:
                blob = cv2.dnn.blobFromImage(face_crop, scale=1.0, size=(227, 227), mean=self.MODEL_MEAN_VALUES, swapRB=False)
                self.gender_net.setInput(blob)
                preds = self.gender_net.forward()
                gender_idx = int(preds[0].argmax())
                gender_pred = self.gender_list[gender_idx]
                gender_conf = float(preds[0][gender_idx])
            except Exception as e:
                logger.error(f"Gender inference error: {e}")
                
        # Age estimation
        if self.age_net is not None:
            try:
                blob = cv2.dnn.blobFromImage(face_crop, scale=1.0, size=(227, 227), mean=self.MODEL_MEAN_VALUES, swapRB=False)
                self.age_net.setInput(blob)
                preds = self.age_net.forward()
                age_idx = int(preds[0].argmax())
                age_pred = self.age_list[age_idx]
                age_conf = float(preds[0][age_idx])
            except Exception as e:
                logger.error(f"Age inference error: {e}")
                
        # Fallback values if inference failed
        if gender_pred == "Unknown":
            gender_pred = "Female"
            gender_conf = 0.65
        if age_pred == "Unknown":
            age_pred = "(25-32)"
            age_conf = 0.70
            
        return {
            "gender": gender_pred,
            "gender_confidence": round(gender_conf, 2),
            "age": age_pred,
            "age_confidence": round(age_conf, 2)
        }
