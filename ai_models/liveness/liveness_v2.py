import cv2
import numpy as np
import logging
from ai_models.liveness.liveness_detector import LivenessDetector

logger = logging.getLogger(__name__)

class LivenessDetectorV2(LivenessDetector):
    def __init__(self, ear_threshold: float = 0.2, consecutive_frames: int = 3):
        super().__init__(ear_threshold, consecutive_frames)
        logger.info("Advanced Liveness & Deepfake V2 Detector successfully loaded.")
        
    def detect_deepfake_or_swap(self, face_crop: np.ndarray) -> dict:
        """
        Detects face-swaps, deepfakes, or digital manipulations by analyzing
        spectral frequency residuals and edge blending inconsistencies.
        
        Deepfakes typically display blurred boundary transitions around eyes, mouth,
        and outer face masks, yielding specific pixel frequency distribution profiles.
        """
        if face_crop is None or face_crop.size == 0:
            return {"is_deepfake": False, "confidence": 0.0}
            
        try:
            # 1. Analyze high-frequency edge blend regions
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            # Resize crop to standard grid
            resized = cv2.resize(gray, (128, 128)).astype(np.float32)

            # Early-exit: uniform / blank frame has zero texture -> definitive synthetic signal
            pixel_std = float(np.std(resized))
            if pixel_std < 1.0:
                return {
                    "is_deepfake": True,
                    "confidence": 1.0,
                    "residual_ratio": 0.0
                }

            # Compute discrete Fourier transform (DFT) to analyze spatial frequencies
            dft = cv2.dft(resized, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            # Calculate magnitude spectrum (all values are >= 0 after magnitude())
            mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
            total_magnitude = float(np.sum(mag)) + 1e-6
            # High-frequency energy in outer quadrant (peripheral edges)
            high_freq_energy = float(np.sum(mag[96:, 96:]))

            # Ratio of high-frequency energy to total spectrum energy
            ratio = high_freq_energy / total_magnitude

            is_deepfake = False
            confidence = 0.0

            if ratio < 0.005:
                # Minimal high-frequency detail -> GAN/Deepfake smoothing signature
                is_deepfake = True
                confidence = round(1.0 - (ratio / 0.005), 3)
            else:
                # Natural face: rich high-frequency texture
                is_deepfake = False
                confidence = round(min(ratio / 0.02, 1.0), 3)
                
            return {
                "is_deepfake": is_deepfake,
                "confidence": min(1.0, confidence),
                "residual_ratio": round(ratio, 4)
            }
        except Exception as e:
            logger.error(f"Deepfake detection pipeline warning: {e}")
            return {"is_deepfake": False, "confidence": 0.5}
            
    def evaluate_liveness_v2(self, face_crop: np.ndarray, left_eye: list = None, right_eye: list = None) -> dict:
        """
        Comprehensive V2 Liveness and Deepfake analysis interface.
        """
        # Call legacy blink & texture routines
        base_res = self.analyze_liveness(face_crop, left_eye, right_eye)
        # Run deepfake residual checks
        deepfake_res = self.detect_deepfake_or_swap(face_crop)
        
        # Combine anomalies into a final security threat score
        liveness_score = base_res["liveness_score"]
        is_spoof = base_res["is_spoof"]
        
        if deepfake_res["is_deepfake"]:
            # Drop liveness score drastically if deepfake detected
            liveness_score = max(0.0, liveness_score - 0.4)
            if liveness_score < 0.5:
                is_spoof = True
                
        return {
            "liveness_score": round(liveness_score, 3),
            "is_spoof": is_spoof,
            "blink_detected": base_res["blink_detected"],
            "total_blinks": base_res["total_blinks"],
            "deepfake_alert": deepfake_res["is_deepfake"],
            "deepfake_confidence": deepfake_res["confidence"]
        }
