import os
import time
import csv
import cv2
import argparse
from pathlib import Path
from detect_landmarks import detect_facial_landmarks  # trả (ear, mar, frame)

# ====== Chuẩn hoá đường dẫn theo cây: code/, data/, music/ ======
HERE = Path(__file__).resolve()
PROJ = HERE.parent.parent
DATA_DIR = PROJ / "data"

# ====== Ghi CSV EAR/MAR ======
def record_csv(out_csv: Path, seconds: int = 15, feature: str = "ear", cam_index: int = 1):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Use the requested camera index (was hard-coded to 1 previously)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERR] Không mở được camera index {cam_index}. Thử index khác (0,1,2...).")
        return

    values = []
    t0 = time.time()
    print(f"[INFO] Bắt đầu ghi '{feature.upper()}' trong ~{seconds}s → {out_csv}")
    print("[INFO] Nhấn 'q' để dừng sớm.")

    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            print("[ERR] Không đọc được khung hình từ camera.")
            break

        ear, mar, vis = detect_facial_landmarks(frame)  # frame BGR vào, trả vis có overlay
        if ear is not None:
            values.append(float(ear if feature == "ear" else mar))

        # HUD
        txt = f"Recording {feature.upper()} | {len(values)} frames"
        cv2.putText(vis, txt, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Record EAR/MAR", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Lưu CSV
    if values:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([feature])
            for v in values:
                w.writerow([v])
        print(f"[OK] Đã lưu {len(values)} dòng vào: {out_csv}")
    else:
        print("[WARN] Không có dữ liệu để lưu (có thể chưa thấy mặt trên camera).")


def parse_args():
    p = argparse.ArgumentParser(description="Ghi CSV EAR/MAR từ webcam")
    # class -> tự suy ra feature
    p.add_argument("--cls", choices=[
        "eyes_natural","eyes_sleepy","eyes_blink",
        "mouth_natural","mouth_open","mouth_yawn"
    ], default="eyes_natural", help="Chọn lớp cần ghi")
    p.add_argument("--seconds", type=int, default=15, help="Thời gian ghi (giây)")
    p.add_argument("--cam", type=int, default=0, help="Camera index (0/1/2...)")
    p.add_argument("--out", type=str, default="", help="Tự chỉ định đường dẫn CSV (tùy chọn)")
    return p.parse_args()


if __name__ == "__main__":
    # Giảm bớt spam log của TF/Mediapipe (không bắt buộc)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    args = parse_args()
    # Map class -> feature & thư mục lưu
    if args.cls.startswith("eyes_"):
        feature = "ear"
        base_dir = DATA_DIR / "eyes" / args.cls
    else:
        feature = "mar"
        base_dir = DATA_DIR / "mouth" / args.cls

    # Tên file mặc định: timestamp.csv nếu không truyền --out
    if args.out:
        out_csv = Path(args.out)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = base_dir / f"{args.cls}_{ts}.csv"

    record_csv(out_csv=out_csv, seconds=args.seconds, feature=feature, cam_index=args.cam)
