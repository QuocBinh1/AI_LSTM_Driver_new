# ui_dashboard.py
import streamlit as st


def init_page():
    """Cấu hình trang + CSS chung."""
    st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
    st.markdown(
        """
        <style>
        .main {
            background-color: #020617;      /* nền tối */
        }
        .block-container {
            padding-top: 10px;
            max-width: 1200px;
        }
        .kh-title {
            font-size: 40px;
            font-weight: 600;
            text-align: center;
            color: #e5e7eb;
            margin-bottom: 14px;
            margin-top: 20px;
        }
        .kh-card {
            background-color: #020617;
            border-radius: 16px;
            padding: 16px 20px;
            border: 1px solid #1e293b;
            box-shadow: 0 8px 30px rgba(15,23,42,0.6);
            margin-bottom: 14px;
           
        }
        .kh-card-title {
            font-weight: 600;
            font-size: 15px;
            margin-bottom: 10px;
            color: #e5e7eb;
        }
        .kh-badge-live {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 500;
            background: #f97316;
            color: #020617;
            margin-bottom: 8px;
        }
        .kh-badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #dc2626;
        }
        .kh-metric-label {
            font-size: 12px;
            color: #9ca3af;
        }
        .kh-metric-value {
            font-size: 20px;
            font-weight: 600;
            color: #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='kh-title'>Hệ Thống Phát Hiện Trạng Thái Buồn Ngủ Của Tài Xế Khi Lái Xe Ô Tô</div>",
        unsafe_allow_html=True,
    )


def build_layout():
    """
    Tạo layout:
    - Bên trái: card Camera
    - Bên phải: card Trạng thái & Thống kê
    Trả về: run, frame_placeholder, status_placeholder, stats_placeholder
    """
    col_left, col_right = st.columns([2.2, 1])

    # ----- CARD CAMERA -----
    with col_left:
        st.markdown("<div class='kh-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='kh-card-title'>📷 Camera giám sát</div>",
            unsafe_allow_html=True,
        )
        # bọc frame vào flex - khung camera ở giữa card
        st.markdown(
            "<div style='display:flex; justify-content:center;'>",
            unsafe_allow_html=True,
        )

        frame_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        # --- NÚT START / STOP ---
        if "run_webcam" not in st.session_state:
            st.session_state.run_webcam = False

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶ Bắt đầu", use_container_width=True):
                st.session_state.run_webcam = True
        with btn_col2:
            if st.button("⏹ Dừng", use_container_width=True):
                st.session_state.run_webcam = False

        # giá trị run trả về cho app.py
        run = st.session_state.run_webcam

        st.markdown("</div>", unsafe_allow_html=True)
            
    # ----- CÁC CARD BÊN PHẢI -----
    with col_right:
        # Trạng thái tài xế
        st.markdown("<div class='kh-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='kh-card-title'>🧍‍♂️ Trạng thái tài xế</div>",
            unsafe_allow_html=True,
        )
        status_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        # Thống kê cảnh báo
        st.markdown("<div class='kh-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='kh-card-title'>📊 Thống kê cảnh báo</div>",
            unsafe_allow_html=True,
        )
        stats_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    return run, frame_placeholder, status_placeholder, stats_placeholder
