# file: telegram_bot.py

import requests
import datetime
import cv2  # Cần để mã hóa ảnh

# --- Thông tin bí mật của Bot ---
TOKEN = "8492461159:AAGm7LOXoqOkoYdzUSxJOdqIJ_N_AQertXg"
CHAT_ID = "6598189722"
# --------------------------------

def send_telegram_alert(message):
    """Gửi cảnh báo CHỈ CÓ CHỮ (dự phòng)"""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message}
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print(f"Lỗi gửi text Telegram: {e}")

def send_telegram_photo_alert(frame, message_text):
    """
    Gửi cảnh báo CÓ ẢNH (frame) và Chú thích (caption).
    """
    # 1. Lấy thời gian hiện tại
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. Tạo caption giống như ảnh mẫu
    caption = f" Cảnh báo: {message_text}\nThời gian: {now_str}"
    
    # 3. Mã hóa frame (ảnh) sang định dạng JPG trong bộ nhớ
    success, img_buffer = cv2.imencode('.jpg', frame)
    if not success:
        print("Lỗi: Không thể mã hóa ảnh.")
        # Nếu lỗi, gửi tạm tin nhắn chữ
        send_telegram_alert(caption) 
        return
    
    # 4. Chuẩn bị file và payload để gửi
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    payload = {
        'chat_id': CHAT_ID,
        'caption': caption
    }
    files = {
        'photo': ('alert_image.jpg', img_buffer.tobytes(), 'image/jpeg')
    }
    
    # 5. Gửi request
    try:
        # Tăng timeout một chút vì gửi ảnh mất thời gian hơn
        requests.post(url, data=payload, files=files, timeout=10)
    except Exception as e:
        print(f"Lỗi khi gửi ảnh Telegram (bỏ qua): {e}")