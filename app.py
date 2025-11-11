import time
from collections import deque

import cv2
import numpy as np
import streamlit as st

from inference import predict_eye, predict_mouth
from detect_landmarks import detect_facial_landmarks
from audio_alert import play_audio, reset_audio_state  # âm thanh cảnh báo
from telegram import send_telegram_photo_alert, send_telegram_alert  # gửi cảnh báo Telegram

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("🚗 Hệ thống phát hiện buồn ngủ cho tài xế (LSTM + EAR/MAR)")
st.write(
    "Ứng dụng sử dụng webcam, trích xuất đặc trưng mắt/miệng (EAR/MAR) từ MediaPipe, "
    "kết hợp mô hình LSTM + heuristic thời gian để phát hiện chớp mắt, buồn ngủ và ngáp. "
    "Có bước hiệu chỉnh tự động 3 giây đầu, hỗ trợ cảnh báo âm thanh và gửi cảnh báo Telegram."
)

# ================== THAM SỐ HỆ THỐNG ==================
SEQ_LEN = 12

# Thời gian & ngưỡng logic
SLEEP_MIN_DUR = 3.0          # mắt nhắm liên tục >= 3s -> buồn ngủ
REFRACTORY_AFTER_OPEN = 0.4  # sau khi mở mắt, 0.4s không báo lại

TALK_LOW = 0.16              # dưới -> đóng; vùng giữa -> nói
MOUTH_OPEN_T = 0.30          # há nhẹ
OPEN_MIN_DUR = 0.50          # há nhẹ >=0.5s
YAWN_FACTOR = 1.4            # YAWN_T = YAWN_FACTOR * MAR_mở
YAWN_MIN_DUR = 1.20          # há to >=1.2s -> YAWN

BASELINE_CALIB_TIME = 3.0    # 3 giây đầu để đo EAR/MAR

# ================== SESSION STATE ==================
s = st.session_state

if "ear_buf" not in s:
    s.ear_buf = deque(maxlen=SEQ_LEN)
    s.mar_buf = deque(maxlen=SEQ_LEN)

# mắt
if "eye_is_closed" not in s:
    s.eye_is_closed = False
    s.eye_closed_since = 0.0
    s.sleepy_active = False
    s.eye_refractory_until = 0.0

# miệng
if "mouth_open_since" not in s:
    s.mouth_open_since = 0.0
    s.mid_open_since = 0.0
    s.prev_mouth_state = "closed"
    s.talk_osc = 0

# baseline & ngưỡng động
if "ear_open_avg" not in s:
    s.ear_open_avg = None
    s.mar_open_avg = None
    s.BLINK_T_CLOSE = 0.32  # fallback
    s.BLINK_T_OPEN = 0.38
    s.YAWN_T = 0.45

# cờ Telegram (tránh spam)
if "sent_drowsy_alert" not in s:
    s.sent_drowsy_alert = False
if "sent_yawn_alert" not in s:
    s.sent_yawn_alert = False

# ================== UI ==================
col_left, col_right = st.columns([2.4, 1])

with col_right:
    run = st.checkbox("▶ Bắt đầu từ webcam", value=False, key="run_webcam")
    st.markdown(
        """
        **Quy trình demo:**
        1. Bật webcam → 3 giây đầu hệ thống tự hiệu chỉnh (mắt mở, miệng bình thường).
        2. Sau đó thử:
           - Nháy mắt bình thường → `BLINK`.
           - Nhắm mắt ≥ 3s → `DROWSY` + âm báo + Telegram.
           - Ngáp to (há rộng & lâu ≥ 1.2s) → `YAWNING` + âm báo + Telegram.
        3. Khi trở lại bình thường → `ALERT`, reset cảnh báo.
        """
    )

frame_placeholder = col_left.empty()
status_placeholder = col_right.empty()

# ================== AUTO CALIBRATION ==================
def auto_calibrate(cap):
    st.info("📷 Đang hiệu chỉnh 3 giây đầu. Nhìn thẳng, mở mắt & ngậm miệng bình thường...")
    ear_vals, mar_vals = [], []
    start = time.time()

    while time.time() - start < BASELINE_CALIB_TIME:
        ok, frame = cap.read()
        if not ok:
            break

        ear, mar, vis = detect_facial_landmarks(frame)
        if ear is not None:
            ear_vals.append(ear)
        if mar is not None:
            mar_vals.append(mar)

        frame_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    # Tính trung bình
    s.ear_open_avg = float(np.mean(ear_vals)) if ear_vals else 0.4
    s.mar_open_avg = float(np.mean(mar_vals)) if mar_vals else 0.25

    # Ngưỡng động theo từng người
    s.BLINK_T_CLOSE = 0.80 * s.ear_open_avg
    s.BLINK_T_OPEN = 0.92 * s.ear_open_avg
    s.YAWN_T = max(YAWN_FACTOR * s.mar_open_avg, 0.4)

    st.success(
        f"Hiệu chỉnh xong ✅ EAR≈{s.ear_open_avg:.3f}, MAR≈{s.mar_open_avg:.3f}\n\n"
        f"- BLINK_T_CLOSE={s.BLINK_T_CLOSE:.3f}\n"
        f"- BLINK_T_OPEN={s.BLINK_T_OPEN:.3f}\n"
        f"- YAWN_T={s.YAWN_T:.3f}"
    )
    time.sleep(1.0)

# ================== MAIN LOOP ==================
if run:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Không mở được webcam. Vui lòng kiểm tra thiết bị.")
    else:
        # Chỉ hiệu chỉnh 1 lần cho phiên
        if s.ear_open_avg is None:
            auto_calibrate(cap)

        while True:
            if not s.run_webcam:
                break

            ok, frame = cap.read()
            if not ok:
                st.error("Không đọc được frame từ webcam.")
                break

            now = time.time()
            ear, mar, vis_frame = detect_facial_landmarks(frame)

            status = "..."
            status_text = "Đang dò khuôn mặt..."
            status_level = "info"
            color = (0, 255, 255)

            blink_flag = False
            yawn_flag_rule = False
            mid_open_active = False
            talk_flag = False

            if ear is not None and mar is not None:
                # buffer cho LSTM
                s.ear_buf.append(float(ear))
                s.mar_buf.append(float(mar))

                # làm mượt nhẹ (3 frame)
                ear_now = float(np.mean(list(s.ear_buf)[-3:]))
                mar_now = float(np.mean(list(s.mar_buf)[-3:]))

                # ===== 1) Heuristic mắt =====
                if s.eye_is_closed:
                    if ear_now > s.BLINK_T_OPEN:
                        dur = now - s.eye_closed_since
                        blink_flag = dur < 0.35
                        s.eye_is_closed = False
                        s.eye_closed_since = 0.0
                        s.sleepy_active = False
                        s.eye_refractory_until = now + REFRACTORY_AFTER_OPEN
                    else:
                        if (now - s.eye_closed_since) >= SLEEP_MIN_DUR:
                            s.sleepy_active = True
                else:
                    if ear_now < s.BLINK_T_CLOSE:
                        s.eye_is_closed = True
                        s.eye_closed_since = now

                # ===== 2) Heuristic miệng =====
                if mar_now >= s.YAWN_T:
                    if s.prev_mouth_state != "wide":
                        s.mouth_open_since = now
                    s.prev_mouth_state = "wide"
                    s.mid_open_since = 0.0
                elif mar_now >= MOUTH_OPEN_T:
                    if s.prev_mouth_state != "mid":
                        s.mid_open_since = now
                    s.prev_mouth_state = "mid"
                elif TALK_LOW <= mar_now < MOUTH_OPEN_T:
                    if s.prev_mouth_state == "closed":
                        s.talk_osc += 1
                    s.prev_mouth_state = "talk"
                    s.mid_open_since = 0.0
                else:
                    if s.prev_mouth_state in ("talk", "wide"):
                        if s.prev_mouth_state == "talk":
                            s.talk_osc += 1
                    s.prev_mouth_state = "closed"
                    s.mid_open_since = 0.0

                # Rule: YAWN (há to lâu)
                if (
                    s.prev_mouth_state == "wide"
                    and (now - s.mouth_open_since) >= YAWN_MIN_DUR
                ):
                    yawn_flag_rule = True
                    s.talk_osc = 0

                # MOUTH OPEN nhẹ lâu
                if (
                    s.prev_mouth_state == "mid"
                    and s.mid_open_since > 0
                    and (now - s.mid_open_since) >= OPEN_MIN_DUR
                ):
                    mid_open_active = True

                # TALKING (dao động nhiều)
                if s.talk_osc >= 3:
                    talk_flag = True
                    s.talk_osc = 0

                # ===== 3) LSTM predict khi đủ chuỗi =====
                eye_label = mouth_label = "..."
                if len(s.ear_buf) == SEQ_LEN:
                    eye_label, _ = predict_eye(list(s.ear_buf))
                if len(s.mar_buf) == SEQ_LEN:
                    mouth_label, _ = predict_mouth(list(s.mar_buf))

                # ===== 4) Kết hợp rule + LSTM =====
                yawn_decide = (
                    (mouth_label == "mouth_yawn" and mar_now >= s.YAWN_T)
                    or yawn_flag_rule
                )
                # chỉ dùng rule mắt (sleepy_active), không để LSTM tự kéo xuống DROWSY
                sleepy_decide = (
                    s.sleepy_active
                    and (now >= s.eye_refractory_until)
                )

                if yawn_decide:
                    status = "YAWNING"
                    status_text = "⚠ Cảnh báo: Ngáp nhiều / há miệng lớn kéo dài."
                    status_level = "warning"
                    color = (0, 165, 255)

                    play_audio("mouth_yawn")
                    if not s.sent_yawn_alert:
                        try:
                            send_telegram_photo_alert(vis_frame, "Phát hiện tài xế ngáp nhiều / há miệng kéo dài.")
                        except Exception as e:
                            print("Telegram YAWN error:", e)
                        s.sent_yawn_alert = True

                elif sleepy_decide:
                    status = "DROWSY"
                    status_text = "⚠ Nguy hiểm: Mắt nhắm lâu, có dấu hiệu buồn ngủ."
                    status_level = "error"
                    color = (0, 0, 255)

                    play_audio("eyes_sleepy")
                    if not s.sent_drowsy_alert:
                        try:
                            send_telegram_photo_alert(vis_frame, "Phát hiện tài xế buồn ngủ, mắt nhắm liên tục.")
                        except Exception as e:
                            print("Telegram DROWSY error:", e)
                        s.sent_drowsy_alert = True

                elif talk_flag:
                    status = "TALKING"
                    status_text = "Đang nói chuyện."
                    status_level = "info"
                    color = (0, 255, 255)
                    reset_audio_state()

                elif mid_open_active:
                    status = "MOUTH OPEN"
                    status_text = "Miệng mở nhẹ trong thời gian dài."
                    status_level = "info"
                    color = (0, 255, 255)
                    reset_audio_state()

                elif blink_flag:
                    status = "BLINK"
                    status_text = "Chớp mắt bình thường."
                    status_level = "success"
                    color = (0, 255, 0)
                    reset_audio_state()

                else:
                    status = "ALERT"
                    status_text = "✅ Tỉnh táo."
                    status_level = "success"
                    color = (0, 255, 0)
                    reset_audio_state()
                    # reset cờ để lần sau vẫn gửi được
                    s.sent_drowsy_alert = False
                    s.sent_yawn_alert = False

                # HUD
                cv2.putText(
                    vis_frame,
                    f"Status: {status}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis_frame,
                    f"EAR:{ear_now:.3f} MAR:{mar_now:.3f}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            else:
                # Không thấy mặt
                status_text = "Không nhận diện được khuôn mặt. Hãy ngồi gần hơn & đủ sáng."
                status_level = "info"
                cv2.putText(
                    vis_frame,
                    "No face detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                reset_audio_state()

            # ===== HIỂN THỊ =====
            frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            if status_level == "error":
                status_placeholder.error(status_text)
            elif status_level == "warning":
                status_placeholder.warning(status_text)
            elif status_level == "success":
                status_placeholder.success(status_text)
            else:
                status_placeholder.info(status_text)

            time.sleep(0.03)

        cap.release()
