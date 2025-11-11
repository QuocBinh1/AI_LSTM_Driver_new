# Tạo .npz chuỗi từ CSV (EAR/MAR) + lưu file_ids để split đúng
# create_sequence_dataset_earmar.py

import glob
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve()
PROJ = HERE.parent.parent
DATA_DIR = PROJ / "data"

SEQ_LEN = 12
STRIDE  = 1

def _windows(a, L=30, S=1):
    X = []
    for i in range(0, len(a) - L + 1, S):
        X.append(a[i:i+L])
    return np.array(X, np.float32)

def build_npz(data_root, classes, out_path, feature_name):
    data_root = Path(data_root)
    X, y, file_ids = [], [], []

    for cls in classes:
        for csvf in sorted((data_root / cls).glob("*.csv")):
            arr = np.loadtxt(csvf, delimiter=",", skiprows=1, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            feat = arr[:, 0]
            win = _windows(feat, L=SEQ_LEN, S=STRIDE)[..., None]  # (n,T,1)
            if len(win) == 0:
                continue

            X.append(win)
            y.append(np.full((len(win),), cls, dtype=object))
            file_ids.append(np.full((len(win),), csvf.name, dtype=object))

    if not X:
        raise RuntimeError(f"No CSV found under {data_root} for classes={classes}")

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    file_ids = np.concatenate(file_ids, axis=0)

    np.savez_compressed(out_path, X=X, y=y, file_ids=file_ids, feature=feature_name)
    print(f"Saved {out_path} -> X{X.shape}, y{y.shape}, file_ids{file_ids.shape}")

if __name__ == "__main__":
    build_npz(DATA_DIR / "eyes",  ["eyes_natural","eyes_sleepy"],
              DATA_DIR / "dataset_sequences_eye.npz",   "ear")
    build_npz(DATA_DIR / "mouth", ["mouth_natural","mouth_yawn"],
              DATA_DIR / "dataset_sequences_mouth.npz", "mar")
