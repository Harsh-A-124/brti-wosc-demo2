import time
import threading

import av
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Coastal Safety Detection",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Mobile-friendly CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1rem 1rem 2rem; }
    video { width: 100% !important; max-height: 70vh; object-fit: contain; }
    .stat-box {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid #4ade80;
        font-size: 1rem;
        color: #e2e8f0;
    }
    .stat-box.danger  { border-left-color: #f87171; }
    .stat-box.warning { border-left-color: #fbbf24; }
    h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "YOLOv8n":       "models/yolov8n.pt",
    "YOLOv8m":   "models/yolov8m.pt",
    "YOLOv11m":  "models/yolov11m.pt",
}

CLASS_COLORS = {
    "Human":       (0, 200, 100),
    "Rip_Current": (0, 60, 255),
}

# Per-class confidence thresholds
CONF_THRESHOLDS = {
    "Human":       0.6,
    "Rip_Current": 0.25,
}

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name: str) -> YOLO:
    return YOLO(MODEL_PATHS[name])

# ── Shared detection state ─────────────────────────────────────────────────────
class DetectionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}
        self.has_danger: bool = False

detection_state = DetectionState()

# ── WebRTC video processor ─────────────────────────────────────────────────────
class VideoProcessor:
    def __init__(self, model: YOLO):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run at the lowest threshold so the model returns all candidates,
        # then filter per class manually below
        results = self.model.predict(
            source=img,
            conf=min(CONF_THRESHOLDS.values()),
            verbose=False,
        )[0]

        counts: dict[str, int] = {}
        has_danger = False

        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label  = self.model.names[cls_id]
                conf_v = float(box.conf[0])

                # Apply per-class threshold
                if conf_v < CONF_THRESHOLDS.get(label, 0.5):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = CLASS_COLORS.get(label, (200, 200, 200))

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # text = f"{label} {conf_v:.0%}"
                text = f"{label}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(img, text, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                counts[label] = counts.get(label, 0) + 1
                if label == "Rip_Current":
                    has_danger = True

        if has_danger:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], 52), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.putText(img, "WARNING: RIP CURRENT",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        with detection_state.lock:
            detection_state.counts     = counts
            detection_state.has_danger = has_danger

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── RTC configuration ──────────────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🌊 Coastal Safety Detector")
st.caption("Real-time Human & Rip Current detection - runs on the server, works on any device.")

model_choice = st.selectbox("Model", list(MODEL_PATHS.keys()))
model = load_model(model_choice)

# ── WebRTC stream ──────────────────────────────────────────────────────────────
ctx = webrtc_streamer(
    key=f"coastal-{model_choice}",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=lambda: VideoProcessor(model),
    media_stream_constraints={
        "video": {
            "width":  {"ideal": 640},
            "height": {"ideal": 480},
            "facingMode": "environment",
        },
        "audio": False,
    },
    async_processing=True,
)

# ── Live stats panel ───────────────────────────────────────────────────────────
if ctx.state.playing:
    st.markdown("---")
    stats_placeholder = st.empty()

    while ctx.state.playing:
        with detection_state.lock:
            counts     = dict(detection_state.counts)
            has_danger = detection_state.has_danger

        with stats_placeholder.container():
            if not counts:
                st.info("👀 Scanning... Point the camera at the beach.")
            else:
                for label, n in counts.items():
                    icon      = "🌀" if label == "Rip_Current" else "🧍"
                    box_class = "danger" if label == "Rip_Current" else "warning"
                    st.markdown(
                        f'<div class="stat-box {box_class}">'
                        f'{icon} <b>{label}</b>: {n} detected'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if has_danger:
                    st.error("🚨 Rip current present - warn swimmers immediately!")
                else:
                    st.success("✅ No rip currents detected.")

        time.sleep(0.5)