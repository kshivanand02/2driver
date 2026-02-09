import streamlit as st
import cv2
import dlib
import numpy as np
import time
from collections import deque

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection System")
st.markdown("Temporal fusion of blink rate, yawn frequency, head posture over sliding windows")

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False
if "prev_state" not in st.session_state:
    st.session_state.prev_state = "ACTIVE"
if "recovery_count" not in st.session_state:
    st.session_state.recovery_count = 0

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

detector, predictor = load_models()

# ---------------- UTILITIES ----------------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_ratio(a, b, c, d, e, f):
    up = euclidean(b, d) + euclidean(c, e)
    down = euclidean(a, f)
    return up / (2.0 * down)

def head_bent_angle(nose, chin):
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    return np.degrees(np.arctan2(dy, dx))

def weighted_ratio(binary_hist, alpha=0.85):
    if len(binary_hist) == 0:
        return 0.0
    w = np.linspace(alpha, 1.0, len(binary_hist))
    return float(np.sum(w * np.array(binary_hist)) / np.sum(w))

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
start = st.sidebar.button("▶ Start Camera")
stop = st.sidebar.button("⏹ Stop Camera")

MOUTH_THRESH = st.sidebar.slider("Mouth Threshold", 12, 40, 25)
HEAD_BENT_THRESH = st.sidebar.slider("Head Angle Threshold", 60, 120, 90)
EAR_THRESH = 0.20

if start: st.session_state.run = True
if stop: st.session_state.run = False

# ---------------- UI ----------------
left_col, right_col = st.columns([2, 1])
with left_col:
    frame_window = st.empty()
with right_col:
    status_box = st.empty()
    ear_box = st.empty()
    mouth_box = st.empty()
    head_box = st.empty()
    yawn_box = st.empty()
    yawn_count_box = st.empty()
    blink_rate_box = st.empty()

# ---------------- TEMPORAL MEMORY ----------------
WINDOW = 20
ear_hist = deque(maxlen=WINDOW)
eye_closed_hist = deque(maxlen=WINDOW)
head_hist = deque(maxlen=WINDOW)
face_hist = deque(maxlen=WINDOW)
blink_events = deque(maxlen=WINDOW)
yawn_events = deque(maxlen=WINDOW)

RECOVERY_FRAMES = 15  # sustained normal frames before ACTIVE

# ---------------- CAMERA LOOP ----------------
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    yawns = 0
    prev_yawn_status = False
    prev_eye_closed = False
    last_t = time.time()

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = max(1e-3, now - last_t)
        last_t = now

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        left_ear = right_ear = mouth_dist = 0.0
        yawn_status = False
        head_status = "Normal"
        face_detected = len(faces) > 0
        head_angle = 0.0
        eye_closed = False

        for face in faces:
            landmarks = predictor(gray, face)

            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            left = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            left_ear = eye_ratio(left[0], left[1], left[2], left[5], left[4], left[3])
            right_ear = eye_ratio(right[0], right[1], right[2], right[5], right[4], right[3])

            ear_avg = (left_ear + right_ear) / 2
            eye_closed = ear_avg < EAR_THRESH

            upper = (landmarks.part(62).x, landmarks.part(62).y)
            lower = (landmarks.part(66).x, landmarks.part(66).y)
            mouth_dist = euclidean(upper, lower)
            yawn_status = mouth_dist > MOUTH_THRESH

            nose = (landmarks.part(30).x, landmarks.part(30).y)
            chin = (landmarks.part(8).x, landmarks.part(8).y)
            head_angle = head_bent_angle(nose, chin)
            head_status = "Head Bent" if head_angle > HEAD_BENT_THRESH else "Normal"

        blink_event = 1 if (prev_eye_closed and not eye_closed) else 0
        prev_eye_closed = eye_closed
        yawn_event = 1 if (prev_yawn_status and not yawn_status) else 0
        prev_yawn_status = yawn_status
        if yawn_event:
            yawns += 1

        ear_hist.append(ear_avg if face_detected else 0)
        eye_closed_hist.append(1 if eye_closed else 0)
        head_hist.append(1 if head_status == "Head Bent" else 0)
        face_hist.append(1 if face_detected else 0)
        blink_events.append(blink_event)
        yawn_events.append(yawn_event)

        fps_est = 1.0 / dt
        blink_rate_fps = (sum(blink_events) / max(1, len(blink_events))) * fps_est
        yawn_rate_win = sum(yawn_events) / max(1, len(yawn_events))
        head_bent_freq = sum(head_hist) / max(1, len(head_hist))

        closed_ratio_w = weighted_ratio(list(eye_closed_hist), alpha=0.85)
        face_lost_ratio_w = weighted_ratio([1 - x for x in face_hist], alpha=0.85)

        # Base classification
        if face_lost_ratio_w > 0.6:
            state_raw = "HIGH"
        elif (
            closed_ratio_w > 0.45 or
            (closed_ratio_w > 0.30 and head_bent_freq > 0.35) or
            (yawn_rate_win > 0.15 and head_bent_freq > 0.35) or
            (blink_rate_fps > 0.35 and closed_ratio_w > 0.25)
        ):
            state_raw = "MEDIUM"
        elif (
            closed_ratio_w > 0.15 or
            yawn_rate_win > 0.05 or
            head_bent_freq > 0.15 or
            blink_rate_fps > 0.20
        ):
            state_raw = "LOW"
        else:
            state_raw = "ACTIVE"

        # ---- Recovery-aware hysteresis ----
        recovered = (
            closed_ratio_w < 0.10 and
            yawn_rate_win < 0.02 and
            head_bent_freq < 0.10 and
            blink_rate_fps < 0.15 and
            face_lost_ratio_w < 0.10
        )

        if recovered:
            st.session_state.recovery_count += 1
        else:
            st.session_state.recovery_count = 0

        prev_state = st.session_state.prev_state
        if prev_state in ["LOW", "MEDIUM", "HIGH"]:
            if recovered and st.session_state.recovery_count >= RECOVERY_FRAMES:
                state = "ACTIVE"
            else:
                if prev_state == "HIGH" and state_raw in ["MEDIUM", "LOW"]:
                    state = "MEDIUM"
                elif prev_state == "MEDIUM" and state_raw == "LOW":
                    state = "LOW"
                elif prev_state == "LOW" and state_raw == "ACTIVE":
                    state = "LOW"
                else:
                    state = state_raw
        else:
            state = state_raw

        st.session_state.prev_state = state

        # UI
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if state == "ACTIVE":
            status_box.success("ACTIVE")
        elif state == "LOW":
            status_box.warning("LOW DROWSINESS")
        elif state == "MEDIUM":
            status_box.markdown(
                "<div style='background-color:#ffc0cb; padding:10px; border-radius:8px; color:black; font-weight:bold;'>MEDIUM DROWSINESS</div>",
                unsafe_allow_html=True
            )
        else:
            status_box.error("HIGH DROWSINESS 🚨")

        ear_box.info(f"Left EAR: {left_ear:.2f} | Right EAR: {right_ear:.2f}")
        mouth_box.info(f"Mouth Distance: {mouth_dist:.2f}")
        head_box.info(f"Head Angle: {head_angle:.1f}°")
        yawn_box.info(f"Yawning: {'YES' if yawn_status else 'NO'}")
        yawn_count_box.warning(f"Yawn Count: {yawns}")
        blink_rate_box.info(f"Blink Rate (events/sec est.): {blink_rate_fps:.2f}")

        time.sleep(0.04)

    cap.release()
else:
    st.info("Click **Start Camera** to begin detection")
