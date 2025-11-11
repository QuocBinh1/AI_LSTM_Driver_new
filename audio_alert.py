from playsound import playsound
import threading
import os

_audio_state = {"eyes_sleepy": False, "mouth_yawn": False}

def _play_mp3_async(file_path):
    try:
        if not os.path.exists(file_path):
            print("File không tồn tại:", file_path)
            return
        playsound(file_path)
    except Exception as e:
        print("Lỗi phát âm:", e)

def play_audio(alert_type):
    global _audio_state
    if not _audio_state.get(alert_type, False):
        _audio_state[alert_type] = True
        file_map = {
            "eyes_sleepy": "music/eyes_sleepy.mp3",
            "mouth_yawn": "music/mouth_yawn.mp3"
        }
        fpath = file_map.get(alert_type)
        if fpath:
            threading.Thread(target=_play_mp3_async, args=(fpath,), daemon=True).start()

def reset_audio_state():
    global _audio_state
    _audio_state = {k: False for k in _audio_state}
