# Train LSTM (EAR/MAR) với Stratified Group Split theo file
import numpy as np, pickle, tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from collections import defaultdict, Counter
from pathlib import Path
import random

def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X, y = d["X"], d["y"]
    file_ids = d["file_ids"] if "file_ids" in d.files else None
    return X, y, file_ids

def build_model(timesteps, n_feat, n_classes):
    m = models.Sequential([
        layers.Input((timesteps, n_feat)),
        layers.GaussianNoise(0.01),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.35),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def stratified_group_split_by_file(y_str, file_ids, test_size=0.2, seed=42):
    """
    Chia theo nhóm = file_ids và vẫn giữ tỷ lệ lớp (stratify).
    Fallback: nếu bất kỳ lớp nào có <=1 file -> dùng stratified split ở cấp cửa sổ.
    """
    import random
    from collections import defaultdict
    rng = random.Random(seed)

    # map file -> class (giả định mỗi CSV thuộc một lớp)
    file2cls = {}
    for cls, fid in zip(y_str, file_ids):
        if fid not in file2cls:
            file2cls[fid] = cls

    # gom danh sách file theo lớp
    cls2files = defaultdict(list)
    for fid, cls in file2cls.items():
        cls2files[cls].append(fid)

    # nếu bất kỳ lớp nào có <=1 file -> fallback cửa sổ (stratified thường)
    if any(len(files) <= 1 for files in cls2files.values()):
        print("[WARN] Ít file/ lớp (<=1). Fallback về stratified split ở cấp cửa sổ.")
        idx = np.arange(len(y_str))
        from sklearn.model_selection import train_test_split
        tr_idx, val_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=y_str
        )
        return tr_idx, val_idx

    # ngược lại: chia theo file, đảm bảo mỗi lớp có ít nhất 1 file vào val
    val_files = set()
    for cls, files in cls2files.items():
        files = list(files)
        rng.shuffle(files)
        k = max(1, round(len(files) * test_size))
        k = min(k, len(files) - 1)  # giữ lại ít nhất 1 file cho train
        val_files.update(files[:k])

    val_mask = np.array([fid in val_files for fid in file_ids], dtype=bool)
    tr_mask  = ~val_mask

    # nếu vì lý do nào đó val chỉ còn 1 lớp -> fallback stratified thường
    if len(set(y_str[val_mask])) < 2 and len(set(y_str)) >= 2:
        print("[WARN] Val chỉ có 1 lớp sau group split. Fallback stratified thường.")
        idx = np.arange(len(y_str))
        from sklearn.model_selection import train_test_split
        tr_idx, val_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=y_str
        )
        return tr_idx, val_idx

    tr_idx = np.where(tr_mask)[0]
    val_idx = np.where(val_mask)[0]
    return tr_idx, val_idx


def train(npz_path, model_out, enc_out, target_timesteps: int = 12):
    X, y_str, file_ids = load_npz(npz_path)

    # Đưa tất cả mẫu về cùng số timestep = target_timesteps bằng cách lấy đoạn cuối
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [num, timesteps, features], got shape={X.shape}")
    if X.shape[1] != target_timesteps:
        if X.shape[1] < target_timesteps:
            raise ValueError(f"Dataset timesteps {X.shape[1]} < target_timesteps {target_timesteps}. Hãy tạo dataset chuẩn 12.")
        X = X[:, -target_timesteps:, :]
        print(f"[INFO] Sliced timesteps to last {target_timesteps}. New shape: {X.shape}")

    le = LabelEncoder(); y = le.fit_transform(y_str)

    if file_ids is not None:
        tr_idx, val_idx = stratified_group_split_by_file(y_str, file_ids, test_size=0.2, seed=42)
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]
    else:
        from sklearn.model_selection import train_test_split
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # In phân bố lớp để kiểm tra
    def dist(y_int):
        c = Counter(y_int.tolist()); tot = len(y_int)
        return {int(k): f"{v} ({v/tot:.2%})" for k,v in c.items()}
    print("[INFO] Train dist:", dist(ytr), " Val dist:", dist(yval))
    print("[INFO] Classes:", list(le.classes_))

    model = build_model(X.shape[1], X.shape[2], len(le.classes_))
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    hist = model.fit(Xtr, ytr, validation_data=(Xval, yval),
                     epochs=60, batch_size=64, callbacks=cb, verbose=1)

    model.save(model_out)
    with open(enc_out, "wb") as f:
        pickle.dump(le, f)
    print("Saved", model_out, enc_out, "classes=", list(le.classes_))

# paths
HERE = Path(__file__).resolve(); PROJ = HERE.parent.parent
DATA_DIR = PROJ / "data"
MODEL_DIR = PROJ / "backend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Train EYE
    train(DATA_DIR / "dataset_sequences_eye.npz",
          MODEL_DIR / "EyeLSTM.h5",
          MODEL_DIR / "label_encoder_eye.pkl",
          target_timesteps=12)

    # Train MOUTH
    train(DATA_DIR / "dataset_sequences_mouth.npz",
          MODEL_DIR / "MouthLSTM.h5",
          MODEL_DIR / "label_encoder_mouth.pkl",
          target_timesteps=12)
