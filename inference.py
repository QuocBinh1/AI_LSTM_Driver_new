# inference.py
from pathlib import Path
import numpy as np
import tensorflow as tf
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "backend" / "models"

SEQ_LEN = 12

# ===== Load model & encoder một lần =====
eye_model = tf.keras.models.load_model(MODEL_DIR / "EyeLSTM.h5")
mouth_model = tf.keras.models.load_model(MODEL_DIR / "MouthLSTM.h5")

with open(MODEL_DIR / "label_encoder_eye.pkl", "rb") as f:
    enc_eye = pickle.load(f)

with open(MODEL_DIR / "label_encoder_mouth.pkl", "rb") as f:
    enc_mouth = pickle.load(f)

EYE_CLASSES = list(enc_eye.classes_)
MOUTH_CLASSES = list(enc_mouth.classes_)

# Hàm chuẩn hoá chuỗi đầu vào theo dạng (1,12,1) , 1 chuỗi, 12 giá trị, 1 đặc trưng duy nhất
def _prep(seq, target_len=SEQ_LEN):
    """Chuẩn hoá chuỗi về (1, target_len, 1). Thiếu thì lặp giá trị cuối."""
    arr = np.asarray(seq, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((target_len,), dtype=np.float32)
    if arr.size < target_len:
        pad = np.full((target_len - arr.size,), float(arr[-1]), dtype=np.float32)
        arr = np.concatenate([arr, pad])
    else:
        arr = arr[-target_len:]
    return arr.reshape(1, target_len, 1)


def predict_eye(ear_seq):
    """Trả về (label, probs) cho mắt."""
    x = _prep(ear_seq, SEQ_LEN)
    probs = eye_model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return EYE_CLASSES[idx], probs


def predict_mouth(mar_seq):
    """Trả về (label, probs) cho miệng."""
    x = _prep(mar_seq, SEQ_LEN)
    probs = mouth_model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return MOUTH_CLASSES[idx], probs
