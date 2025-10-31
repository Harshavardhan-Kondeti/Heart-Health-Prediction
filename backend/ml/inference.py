import os
import numpy as np
import joblib
from typing import Any, Dict, Optional, Tuple, List

ONNXRuntime = None
try:
    import onnxruntime as ort  # type: ignore
    ONNXRuntime = ort
except Exception:
    ONNXRuntime = None

TF = None
try:
    import tensorflow as tf  # type: ignore
    TF = tf
except Exception:
    TF = None


def _load_class_names() -> Optional[List[str]]:
    path = os.getenv("CLASS_NAMES_PATH", "models/class_names.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        return names if names else None
    except Exception:
        return None


class ModelService:
    def __init__(self):
        self.onnx_path = os.getenv("MODEL_ONNX_PATH", "models/model.onnx")
        self.pkl_path = os.getenv("MODEL_PKL_PATH", "models/model.pkl")
        self.keras_path = os.getenv("MODEL_KERAS_PATH", "models/ecg_cnn_effb0.keras")
        self.default_img_size = int(os.getenv("MODEL_IMAGE_SIZE", "380"))
        self.normal_class_index = int(os.getenv("NORMAL_CLASS_INDEX", "3"))
        self.class_names = _load_class_names()
        self._onnx_session = None
        self._pkl_model = None
        self._tf_model = None
        self._load_model()

    def _load_model(self):
        if TF is not None and os.path.exists(self.keras_path):
            try:
                self._tf_model = TF.keras.models.load_model(self.keras_path)
                return
            except Exception:
                pass
        if os.path.exists(self.onnx_path) and ONNXRuntime is not None:
            try:
                self._onnx_session = ONNXRuntime.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
                return
            except Exception:
                self._onnx_session = None
        if os.path.exists(self.pkl_path):
            try:
                self._pkl_model = joblib.load(self.pkl_path)
                return
            except Exception:
                self._pkl_model = None

    def get_expected_image_size(self) -> Optional[Tuple[int, int]]:
        if self._tf_model is None:
            return (self.default_img_size, self.default_img_size)
        try:
            shape = self._tf_model.input_shape  # type: ignore[attr-defined]
            if isinstance(shape, list):
                shape = shape[0]
            if shape is not None and len(shape) == 4:
                _, h, w, c = shape
                if isinstance(h, int) and isinstance(w, int):
                    return (h, w)
        except Exception:
            pass
        return (self.default_img_size, self.default_img_size)

    def _prepare_image_for_keras(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            x = image.astype(np.float32)[None, ...]
        elif image.ndim == 4:
            x = image.astype(np.float32)
        else:
            x = image.astype(np.float32).reshape(1, *image.shape)
        try:
            if TF is not None:
                preprocess = TF.keras.applications.efficientnet.preprocess_input
                x = preprocess(x)
        except Exception:
            pass
        return x

    def _prepare_ecg_for_keras(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 1:
            x = signal.astype(np.float32)
            if np.std(x) > 0:
                x = (x - np.mean(x)) / (np.std(x) + 1e-8)
            x = x.reshape(1, -1, 1)
            return x
        if signal.ndim == 2:
            if signal.shape[0] == 1 or signal.shape[1] == 1:
                x = signal.astype(np.float32).reshape(1, -1, 1)
                return x
            x = signal[:, 0].astype(np.float32).reshape(1, -1, 1)
            return x
        return signal.astype(np.float32).reshape(1, -1, 1)

    def _binary_from_label(self, top_idx: int, top_label: Optional[str]) -> str:
        if top_label is not None and isinstance(top_label, str):
            if "normal" in top_label.lower():
                return "Normal"
        # Fallback to index check
        return "Normal" if top_idx == self.normal_class_index else "Abnormal"

    def predict(self, input_array: np.ndarray) -> Dict[str, Any]:
        if self._tf_model is not None:
            if input_array.ndim in (3, 4):
                X = self._prepare_image_for_keras(input_array)
            else:
                X = self._prepare_ecg_for_keras(input_array)
            preds = self._tf_model.predict(X, verbose=0)
            y = np.asarray(preds)

            if y.ndim == 2 and y.shape[0] == 1 and y.shape[1] > 1:
                probs = y[0]
                top_idx = int(np.argmax(probs))
                top_prob = float(probs[top_idx])
                if self.class_names and top_idx < len(self.class_names):
                    top_label = self.class_names[top_idx]
                else:
                    top_label = None
                binary = self._binary_from_label(top_idx, top_label)
                return {
                    "label": (top_label if top_label is not None else str(top_idx)),
                    "top_label": (top_label if top_label is not None else str(top_idx)),
                    "score": top_prob,
                    "probs": { (self.class_names[i] if self.class_names and i < len(self.class_names) else str(i)): float(probs[i]) for i in range(len(probs)) },
                    "binary": binary,
                    "top_index": top_idx,
                }
            if y.ndim == 2 and y.shape[1] == 1:
                score = float(y[0, 0])
                label = int(score >= 0.5)
                return {"label": label, "score": score, "binary": "Abnormal" if label == 1 else "Normal"}
            s = np.squeeze(y)
            score = float(s) if s.shape == () else float(np.max(s))
            label = int(score >= 0.5)
            return {"label": label, "score": score, "binary": "Abnormal" if label == 1 else "Normal"}

        if self._onnx_session is not None:
            if input_array.ndim == 1:
                X = input_array.reshape(1, -1)
            elif input_array.ndim == 2:
                X = input_array
            else:
                X = input_array.reshape(1, -1)
            input_name = self._onnx_session.get_inputs()[0].name
            preds = self._onnx_session.run(None, {input_name: X.astype(np.float32)})
            y_score = float(np.squeeze(preds[0])) if isinstance(preds, list) else float(np.squeeze(preds))
            label = int(y_score >= 0.5)
            return {"label": label, "score": y_score, "binary": "Abnormal" if label == 1 else "Normal"}

        if self._pkl_model is not None:
            if input_array.ndim == 1:
                X = input_array.reshape(1, -1)
            elif input_array.ndim == 2:
                X = input_array
            else:
                X = input_array.reshape(1, -1)
            model = self._pkl_model
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                score = float(proba[0, 1])
                label = int(score >= 0.5)
                return {"label": label, "score": score, "binary": "Abnormal" if label == 1 else "Normal"}
            else:
                label = int(model.predict(X)[0])
                return {"label": label, "score": None, "binary": "Abnormal" if label == 1 else "Normal"}

        return {
            "label": 0,
            "score": 0.0,
            "binary": "Normal",
            "warning": "Model file not found. Place ecg_cnn_effb0.keras in models/ or set MODEL_KERAS_PATH."
        }


class PPGImageService:
    def __init__(self):
        # Fixed paths to match repository structure
        self.model_path = os.getenv("PPG_MODEL_PATH", "models/PPG_Model/ppg_health_model.pkl")
        self.pca_path = os.getenv("PPG_PCA_PATH", "models/PPG_Model/ppg_pca_transformer.pkl")
        self._model = None
        self._pca = None
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.model_path):
                self._model = joblib.load(self.model_path)
        except Exception:
            self._model = None
        try:
            if os.path.exists(self.pca_path):
                self._pca = joblib.load(self.pca_path)
        except Exception:
            self._pca = None

    def is_ready(self) -> bool:
        return (self._model is not None) and (self._pca is not None)

    def expected_num_features(self) -> Optional[int]:
        try:
            return int(getattr(self._pca, "n_features_in_", None)) if self._pca is not None else None
        except Exception:
            return None

    def preprocess_image_array(self, image: np.ndarray) -> np.ndarray:
        # Convert RGB to grayscale if needed
        if image.ndim == 3 and image.shape[2] == 3:
            # luminosity method
            r = image[..., 0]
            g = image[..., 1]
            b = image[..., 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        elif image.ndim == 2:
            gray = image
        else:
            gray = image.squeeze()

        gray = gray.astype(np.float32)
        # Normalize to 0-1 if looks like 0-255
        if gray.max() > 1.5:
            gray = gray / 255.0

        flat = gray.flatten()

        # Pad/trim to PCA expected length
        n_feat = self.expected_num_features()
        if n_feat is not None:
            if flat.size > n_feat:
                flat = flat[:n_feat]
            elif flat.size < n_feat:
                flat = np.pad(flat, (0, n_feat - flat.size), mode="constant")
        return flat.astype(np.float32)

    def predict_from_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_ready():
            return {
                "label": None,
                "score": None,
                "binary": None,
                "warning": "PPG model/PCA missing. Ensure files exist in models/PPG_Model/.",
            }

        x_vec = self.preprocess_image_array(image)
        # Apply PCA
        try:
            x_pca = self._pca.transform(x_vec.reshape(1, -1))  # type: ignore[attr-defined]
        except Exception:
            return {
                "label": None,
                "score": None,
                "binary": None,
                "warning": "Failed to transform with PCA. Check PCA compatibility.",
            }

        # Predict probability of MI (class 1)
        try:
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(x_pca)  # type: ignore[attr-defined]
                score = float(proba[0, 1])
            else:
                pred = int(self._model.predict(x_pca)[0])  # type: ignore[attr-defined]
                score = float(pred)
        except Exception:
            return {
                "label": None,
                "score": None,
                "binary": None,
                "warning": "Failed during PPG model prediction.",
            }

        label = "MI" if score >= 0.5 else "Normal"
        return {
            "label": label,
            "score": score,
            "binary": label,
        }