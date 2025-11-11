import cv2, numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Chỉ số landmark (theo MediaPipe)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]    # outer-left, upper1, upper2, outer-right, lower1, lower2
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_LR_TB = [78, 308, 13, 14]              # left, right, top, bottom

def _euclid(a, b):
    return float(np.linalg.norm(a - b))

def eye_aspect_ratio(pts, eye_idx):
    """Tính EAR cho 1 mắt"""
    eye = pts[eye_idx]
    A = _euclid(eye[1], eye[5])
    B = _euclid(eye[2], eye[4])
    C = _euclid(eye[0], eye[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

def mouth_aspect_ratio(pts):
    l, r, t, b = MOUTH_LR_TB
    horiz = _euclid(pts[l], pts[r])
    vert  = _euclid(pts[t], pts[b])
    mar = vert / (horiz + 1e-6)
    return mar

def detect_facial_landmarks(frame_bgr):
    """Trích xuất EAR và MAR từ ảnh BGR"""
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not res.multi_face_landmarks:
        return None, None, frame

    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

    # EAR = trung bình 2 mắt
    left_ear  = eye_aspect_ratio(pts, LEFT_EYE)
    right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
    ear = (left_ear + right_ear) / 2.0

    # MAR
    mar = mouth_aspect_ratio(pts)

    # Vẽ hiển thị debug
    cv2.putText(frame, f"EAR: {ear:.3f}  MAR: {mar:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Vẽ contour 2 mắt và miệng cho trực quan
    for i in LEFT_EYE + RIGHT_EYE + MOUTH_LR_TB:
        x, y = int(pts[i][0]), int(pts[i][1])
        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    return ear, mar, frame
