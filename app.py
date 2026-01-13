import time
from collections import deque

import cv2 # OpenCV
import numpy as np
import streamlit as st

from inference import predict_eye, predict_mouth
from detect_landmarks import detect_facial_landmarks
from audio_alert import play_audio, reset_audio_state
from telegram import send_telegram_photo_alert
from ui_dashboard import init_page, build_layout  

# ================== CẤU HÌNH TRANG + LAYOUT ==================
init_page()

#    trạng thái tài xế, thống kê cảnh báo, v.v.
run, frame_placeholder, status_placeholder, stats_placeholder = build_layout()

SEQ_LEN = 12 

LSTM_PRED_BUF_LEN = 5     # chứa 5 lần dự đoán gần nhất [NORMAL, YAWN, YAWN, YAWN, NORMAL]
LSTM_PRED_MIN_VOTES = 3      

SLEEP_MIN_DUR = 3.0          # mắt nhắm liên tục >= 3s -> buồn ngủ
REFRACTORY_AFTER_OPEN = 0.4  # sau khi mở mắt, 0.4s không báo lại

TALK_LOW = 0.16 

MOUTH_OPEN_T = 0.30
OPEN_MIN_DUR = 0.50

YAWN_FACTOR = 1.4  # ngưỡng ngáp 
YAWN_MIN_DUR = 1.20 #  miệng ngáp duy trì >= 1.2s

BASELINE_CALIB_TIME = 3.0 # thời gian hiệu chuẩn ban đầu

BLINK_MIN_DUR = 0.04 # nháy mắt đóng từ 0.04s trở lên
BLINK_MAX_DUR = 0.25 # nháy mắt đóng tối đa 0.25s

# ================== SESSION STATE ==================
s = st.session_state #khởi tạo bộ nhớ trạng thái

# kiểm tra xem biến ear_buffer có chưa , nếu chưa thì tạo 
if "ear_buf" not in s:
    s.ear_buf = deque(maxlen=SEQ_LEN)
    s.mar_buf = deque(maxlen=SEQ_LEN)
    
    s.eye_pred_buf = deque(maxlen=LSTM_PRED_BUF_LEN)
    s.mouth_pred_buf = deque(maxlen=LSTM_PRED_BUF_LEN)

# mắt
if "eye_is_closed" not in s:
    s.eye_is_closed = False #khởi tạo trạng thái mắt
    s.eye_closed_since = 0.0   # thời gian nhắm mắt ,=< 0.3s là nháy ,  3s thì buồn ngủ , 
    s.sleepy_active = False # cờ buồn ngủ
    s.eye_refractory_until = 0.0    # thời gian miễn nhiễm sau khi mở mắt
    s.eye_min_ear_during_close = 1.0  # track EAR thấp nhất trong lần đóng

# miệng
if "mouth_open_since" not in s:
    s.mouth_open_since = 0.0 # thời gian miệng mở to
    s.mid_open_since = 0.0 # thời gian miệng mở nhẹ
    s.prev_mouth_state = "closed" # trạng thái miệng trước đó
    s.talk_osc = 0 # bộ đếm trạng thái nói chuyện

# baseline & ngưỡng động
if "ear_open_avg" not in s:
    s.ear_open_avg = None #trung bình EAR khi mở mắt
    s.mar_open_avg = None #trung bình MAR khi miệng bình thường
    s.BLINK_T_CLOSE = 0.32
    s.BLINK_T_OPEN = 0.38
    s.YAWN_T = 0.45

# cờ Telegram
if "sent_drowsy_alert" not in s:
    s.sent_drowsy_alert = False
if "sent_yawn_alert" not in s:
    s.sent_yawn_alert = False

# thống kê alert
if "total_alerts" not in s:
    s.total_alerts = 0
if "sleepy_events" not in s:
    s.sleepy_events = 0        # buồn ngủ (ngáp)
if "danger_events" not in s:
    s.danger_events = 0        # nguy hiểm (mắt nhắm lâu)
if "prev_status" not in s:
    s.prev_status = "INIT"

# ================== HÀM AUTO CALIBRATION ==================
#mỗi lần cảnh báo xong thì reset trạng thái
def reset_all_states_except_calibration():
    """Reset tất cả state trừ giá trị hiệu chuẩn (ear_open_avg, mar_open_avg)"""
    # Reset buffers
    s.ear_buf.clear()
    s.mar_buf.clear()
    s.eye_pred_buf.clear()
    s.mouth_pred_buf.clear()
    
    # Reset mắt
    s.eye_is_closed = False
    s.eye_closed_since = 0.0
    s.sleepy_active = False
    s.eye_refractory_until = 0.0
    
    # Reset miệng
    s.mouth_open_since = 0.0
    s.mid_open_since = 0.0
    s.prev_mouth_state = "closed"
    s.talk_osc = 0
    
    # Reset flags alert
    s.sent_drowsy_alert = False
    s.sent_yawn_alert = False


def auto_calibrate(cap):
    st.info("Đang hiệu chỉnh 3 giây đầu. Nhìn thẳng, mở mắt & ngậm miệng bình thường...")
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

    # tính trung bình
    s.ear_open_avg = float(np.mean(ear_vals)) if ear_vals else 0.4
    s.mar_open_avg = float(np.mean(mar_vals)) if mar_vals else 0.25

    # ngưỡng động
    s.BLINK_T_CLOSE = 0.80 * s.ear_open_avg
    s.BLINK_T_OPEN = 0.92 * s.ear_open_avg
    s.YAWN_T = max(YAWN_FACTOR * s.mar_open_avg, 0.4)

    st.success(
        f"Hiệu chỉnh xong EAR≈{s.ear_open_avg:.3f}, MAR≈{s.mar_open_avg:.3f}\n\n"
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

            # ===== XỬ LÝ FRAME ===== 
            # trích xuất EAR, MAR từ mediapipe facemesh
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

                #lấy trung bình 3 giá trị gần nhất để giảm nhiễu
                ear_now = float(np.mean(list(s.ear_buf)[-3:]))
                mar_now = float(np.mean(list(s.mar_buf)[-3:]))

                # ===== 1) Heuristic mắt =====
                if s.eye_is_closed:
                    # Track EAR thấp nhất trong lần đóng
                    s.eye_min_ear_during_close = min(s.eye_min_ear_during_close, ear_now)
                    
                    if ear_now > s.BLINK_T_OPEN:
                        dur = now - s.eye_closed_since
                        # Nháy: đóng ngắn + EAR xuống rất thấp (< 0.15, ngưỡng khá hạt sạch)
                        if (dur <= BLINK_MAX_DUR and dur >= BLINK_MIN_DUR and 
                            s.eye_min_ear_during_close < 0.15):
                            blink_flag = True
                            s.sleepy_active = False
                        else:
                            # Nhắm: đóng lâu hoặc EAR không xuống đủ thấp -> buồn ngủ
                            if dur >= SLEEP_MIN_DUR:
                                s.sleepy_active = True

                        # Mắt đã mở lại -> reset trạng thái đóng
                        s.eye_is_closed = False
                        s.eye_closed_since = 0.0
                        s.eye_refractory_until = now + REFRACTORY_AFTER_OPEN
                        s.eye_min_ear_during_close = 1.0  # reset min
                    else:
                        if (now - s.eye_closed_since) >= SLEEP_MIN_DUR:
                            s.sleepy_active = True
                else:
                    if ear_now < s.BLINK_T_CLOSE:
                        s.eye_is_closed = True
                        s.eye_closed_since = now
                        s.eye_min_ear_during_close = ear_now  # bắt đầu track

                # ===== 2) Heuristic miệng =====
                #ngáp wide(mở to), mid(mở nhẹ)
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

                # rule: ngáp
                if (
                    s.prev_mouth_state == "wide"
                    and (now - s.mouth_open_since) >= YAWN_MIN_DUR
                ):
                    yawn_flag_rule = True
                    s.talk_osc = 0

                # mở nhẹ lâu
                if (
                    s.prev_mouth_state == "mid"
                    and s.mid_open_since > 0
                    and (now - s.mid_open_since) >= OPEN_MIN_DUR
                ):
                    mid_open_active = True

                # talking
                if s.talk_osc >= 3:
                    talk_flag = True
                    s.talk_osc = 0

                # ===== 3) LSTM + Smoothing =====
                eye_label = mouth_label = "..."
                if len(s.ear_buf) == SEQ_LEN:
                    eye_label, _ = predict_eye(list(s.ear_buf))
                    s.eye_pred_buf.append(eye_label)
                if len(s.mar_buf) == SEQ_LEN:
                    mouth_label, _ = predict_mouth(list(s.mar_buf))
                    s.mouth_pred_buf.append(mouth_label)

                # Majority vote từ smoothing buffer
                def get_majority_label(buf, default="..."):
                    """Trả về nhãn phổ biến nhất. Nếu không đủ vote trả về default."""
                    if len(buf) < LSTM_PRED_MIN_VOTES:
                        return default
                    from collections import Counter
                    counts = Counter(buf)
                    most_common_label, vote_count = counts.most_common(1)[0]
                    if vote_count >= LSTM_PRED_MIN_VOTES:
                        return most_common_label
                    return default

                # Sử dụng phương pháp bỏ phiếu đa số, cho đến khi bộ đệm có đủ phiếu bầu,
                eye_label_smooth = get_majority_label(s.eye_pred_buf)
                mouth_label_smooth = get_majority_label(s.mouth_pred_buf)

                # ===== 4) KẾT HỢP quyết định cuối , ngưỡng+thời gian và LSTM(dự đoán chuỗi) =====
                yawn_decide = (
                    (mouth_label_smooth == "mouth_yawn" and mar_now >= s.YAWN_T)
                    or yawn_flag_rule
                )
                sleepy_decide = (
                    s.sleepy_active and (now >= s.eye_refractory_until)
                )

                if yawn_decide:
                    status = "YAWNING"
                    status_text = "Cảnh báo: Ngáp nhiều / há miệng lớn kéo dài."
                    status_level = "warning"
                    color = (0, 165, 255)

                    play_audio("mouth_yawn")
                    if not s.sent_yawn_alert:
                        try:
                            send_telegram_photo_alert(
                                vis_frame,
                                "Phát hiện tài xế ngáp nhiều / há miệng kéo dài.",
                            )
                        except Exception as e:
                            print("Telegram YAWN error:", e)
                        s.sent_yawn_alert = True
                        # Reset toàn bộ state sau khi gửi alert (giữ lại calibration)
                        reset_all_states_except_calibration()

                elif sleepy_decide:
                    status = "DROWSY"
                    status_text = "⚠ Nguy hiểm: Mắt nhắm lâu, có dấu hiệu buồn ngủ."
                    status_level = "error"
                    color = (0, 0, 255)

                    play_audio("eyes_sleepy")
                    if not s.sent_drowsy_alert:
                        try:
                            send_telegram_photo_alert(
                                vis_frame,
                                "Phát hiện tài xế buồn ngủ, mắt nhắm liên tục.",
                            )
                        except Exception as e:
                            print("Telegram DROWSY error:", e)
                        s.sent_drowsy_alert = True
                        # Reset toàn bộ state sau khi gửi alert (giữ lại calibration)
                        reset_all_states_except_calibration()

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
                    status = "NORMAL"
                    status_text = "Bình thường - Tỉnh táo."
                    status_level = "success"
                    color = (0, 255, 0)
                    reset_audio_state()

                # cập nhật thống kê
                if status != s.prev_status:
                    if status == "YAWNING":
                        s.total_alerts += 1
                        s.sleepy_events += 1
                    elif status == "DROWSY":
                        s.total_alerts += 1
                        s.danger_events += 1
                    s.prev_status = status

                # HUD trên frame
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

            else:
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

            # trạng thái text
            if status_level == "error":
                status_placeholder.error(status_text)
            elif status_level == "warning":
                status_placeholder.warning(status_text)
            elif status_level == "success":
                status_placeholder.success(status_text)
            else:
                status_placeholder.info(status_text)

            # thống kê cảnh báo
            with stats_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Tổng cảnh báo", s.total_alerts)
                c2.metric("Buồn ngủ", s.sleepy_events)
                c3.metric("Nguy hiểm", s.danger_events)

            time.sleep(0.03)

        cap.release()
