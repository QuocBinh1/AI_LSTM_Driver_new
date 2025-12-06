# Train LSTM (EAR/MAR) với Stratified Group Split theo file
import numpy as np, pickle, tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from collections import defaultdict, Counter
from pathlib import Path
import random
import matplotlib.pyplot as plt
from pathlib import Path


def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X, y = d["X"], d["y"]
    file_ids = d["file_ids"] if "file_ids" in d.files else None
    return X, y, file_ids


def build_model(timesteps, n_feat, n_classes):
    """LSTM 1 tầng + Dense"""
    model = models.Sequential(
        [
            layers.Input(shape=(timesteps, n_feat)),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(32, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def stratified_group_split_by_file(y_str, file_ids, test_size=0.2, seed=42):
    """
    Chia train/val sao cho:
      - Mỗi file_id chỉ nằm trong 1 tập (train hoặc val)
      - Tỉ lệ class trong val gần giống tổng thể
    """
    file_ids = np.array(file_ids)
    y_str = np.array(y_str)

    # Gom nhãn theo file
    file_to_labels = defaultdict(list)
    for fid, lab in zip(file_ids, y_str):
        file_to_labels[fid].append(lab)

    rng = random.Random(seed)
    unique_files = list(file_to_labels.keys())
    rng.shuffle(unique_files)

    # Đếm số mẫu mỗi file
    file_counts = {fid: len(file_to_labels[fid]) for fid in unique_files}
    total_samples = sum(file_counts.values())
    target_val = int(total_samples * test_size)

    val_files = []
    running_val = 0
    for fid in unique_files:
        if running_val < target_val:
            val_files.append(fid)
            running_val += file_counts[fid]

    val_files = set(val_files)

    val_idx = [i for i, fid in enumerate(file_ids) if fid in val_files]
    tr_idx = [i for i, fid in enumerate(file_ids) if fid not in val_files]

    # Debug tỉ lệ label
    def label_ratio(idxs, name):
        c = Counter(y_str[idxs])
        total = len(idxs)
        print(f"[INFO] {name} size={total}")
        for k, v in c.items():
            print(" ", k, f"{v} ({v/total:.2f})")

    label_ratio(tr_idx, "Train")
    label_ratio(val_idx, "Val")

    return np.array(tr_idx), np.array(val_idx)


def train(npz_path, model_out, enc_out, target_timesteps: int = 12):
    X, y_str, file_ids = load_npz(npz_path)

    # Đưa tất cả mẫu về cùng số timestep = target_timesteps bằng cách lấy đoạn cuối
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [num, timesteps, features], got shape={X.shape}")
    if X.shape[1] != target_timesteps:
        if X.shape[1] < target_timesteps:
            raise ValueError(
                f"Dataset timesteps {X.shape[1]} < ...target_timesteps {target_timesteps}. Hãy tạo dataset chuẩn 12."
            )
        X = X[:, -target_timesteps:, :]
        print(f"[INFO] Sliced timesteps to last {target_timesteps}. New shape: {X.shape}")

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    if file_ids is not None:
        tr_idx, val_idx = stratified_group_split_by_file(
            y_str, file_ids, test_size=0.2, seed=42
        )
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]
    else:
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    print("[INFO] Train shape:", Xtr.shape, "Val shape:", Xval.shape)

    timesteps = Xtr.shape[1]
    n_feat = Xtr.shape[2]
    n_classes = len(le.classes_)

    model = build_model(timesteps, n_feat, n_classes)
    model.summary()

    # Callback: EarlyStopping + ModelCheckpoint
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]

    hist = model.fit(
        Xtr,
        ytr,
        validation_data=(Xval, yval),
        epochs=60,
        batch_size=64,
        callbacks=cb,
        verbose=1,
    )
    save_history_plots(hist, model_out)
    compute_and_log_metrics(model, Xtr, ytr, Xval, yval, le, model_out)

    model.save(model_out)
    with open(enc_out, "wb") as f:
        pickle.dump(le, f)
    print("Saved", model_out, enc_out, "classes=", list(le.classes_))


def save_history_plots(hist, model_out):
    """
    Lưu 2 hình:
      - <tên_model>_accuracy.png
      - <tên_model>_loss.png
    cùng thư mục với file .h5
    """
    history = hist.history
    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    if not acc or not loss:
        print("[WARN] History không có accuracy/loss, không vẽ được.")
        return

    model_out = Path(model_out)
    base = model_out.with_suffix("")  # EyeLSTM.h5 -> EyeLSTM
    acc_path = base.parent / f"{base.name}_accuracy.png"
    loss_path = base.parent / f"{base.name}_loss.png"

    epochs = range(1, len(acc) + 1)

    # ----- Hình Accuracy -----
    plt.figure()
    plt.plot(epochs, acc, label="Train accuracy")
    if val_acc:
        plt.plot(epochs, val_acc, label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy - {base.name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # ----- Hình Loss -----
    plt.figure()
    plt.plot(epochs, loss, label="Train loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss - {base.name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    print("[INFO] Đã lưu hình:", acc_path, "và", loss_path)


def compute_and_log_metrics(model, Xtr, ytr, Xval, yval, le, model_out):
    """
    Tính và ghi ra:
      - Loss / Accuracy trên tập train và val
      - Precision / Recall / F1 (macro) trên tập val
      - Confusion matrix
    Đồng thời lưu vào file <model_name>_metrics.txt để dùng cho báo cáo.
    """
    # Đánh giá train / val
    train_loss, train_acc = model.evaluate(Xtr, ytr, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, yval, verbose=0)

    # Dự đoán trên tập val
    y_pred_proba = model.predict(Xval, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Precision / Recall / F1 (macro)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yval, y_pred, average="macro", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(yval, y_pred)

    print("\n========== METRICS ==========")
    print("Classes:", list(le.classes_))
    print(f"Train  - Loss: {train_loss:.4f}  Accuracy: {train_acc:.4f}")
    print(f"Val    - Loss: {val_loss:.4f}  Accuracy: {val_acc:.4f}")
    print(f"Val Precision (macro): {precision:.4f}")
    print(f"Val Recall    (macro): {recall:.4f}")
    print(f"Val F1-score  (macro): {f1:.4f}")
    print("Confusion matrix (rows = y_true, cols = y_pred):")
    print(cm)

    # Lưu ra file txt cạnh file .h5
    model_out = Path(model_out)
    base = model_out.with_suffix("")  # EyeLSTM.h5 -> EyeLSTM
    metrics_path = base.parent / f"{base.name}_metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Classes: " + ", ".join(map(str, le.classes_)) + "\n")
        f.write(f"Train - Loss: {train_loss:.6f}, Accuracy: {train_acc:.6f}\n")
        f.write(f"Val   - Loss: {val_loss:.6f}, Accuracy: {val_acc:.6f}\n")
        f.write(f"Val Precision (macro): {precision:.6f}\n")
        f.write(f"Val Recall    (macro): {recall:.6f}\n")
        f.write(f"Val F1-score  (macro): {f1:.6f}\n\n")

        f.write("Confusion matrix (rows = y_true, cols = y_pred)\n")
        f.write("    " + "\t".join([str(c) for c in le.classes_]) + "\n")
        for i, row in enumerate(cm):
            f.write(str(le.classes_[i]) + "\t" + "\t".join(map(str, row)) + "\n")

        f.write("\nClassification report:\n")
        f.write(
            classification_report(
                yval, y_pred, target_names=le.classes_, digits=4
            )
        )

    print("[INFO] Đã lưu metrics vào:", metrics_path)


# paths
HERE = Path(__file__).resolve()
PROJ = HERE.parent.parent
DATA_DIR = PROJ / "data"
MODEL_DIR = PROJ / "backend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Train EYE
    train(
        DATA_DIR / "dataset_sequences_eye.npz",
        MODEL_DIR / "EyeLSTM.h5",
        MODEL_DIR / "label_encoder_eye.pkl",
        target_timesteps=12,
    )

    # Train MOUTH
    train(
        DATA_DIR / "dataset_sequences_mouth.npz",
        MODEL_DIR / "MouthLSTM.h5",
        MODEL_DIR / "label_encoder_mouth.pkl",
        target_timesteps=12,
    )
