import cv2
import mediapipe as mp
import numpy as np
import threading
import math
import time
import sounddevice as sd
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------- CONFIG ----------------
STATE_FILE = "F:/ModrinthApp/profiles/BCSMP_s7_2.3.0/figura/avatars/FemDex/state.json"
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8080

VOLUME_THRESHOLD_START = 5e-7
VOLUME_THRESHOLD_STOP = 2e-6

# ---------------- GLOBAL STATE ----------------
LATEST_STATE = {
    "emotion": "neutral",
    "speaking": False,
    "volume": 0.0,
    "head_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "pupil": {"x": 0.0, "y": 0.0},
    "eye_open": 1.0,
    "brows": 0.0
}
state_lock = threading.Lock()

# ---------------- AUDIO SETUP ----------------
volume_level = 0.0
speaking = False
def audio_callback(indata, frames, time_, status):
    global volume_level, speaking
    volume_norm = float(np.linalg.norm(indata) / frames)
    volume_level = volume_norm
    if speaking:
        if volume_norm < VOLUME_THRESHOLD_STOP:
            speaking = False
    else:
        if volume_norm > VOLUME_THRESHOLD_START:
            speaking = True

try:
    stream = sd.InputStream(callback=audio_callback)
    stream.start()
except Exception as e:
    print(f"[audio] Failed: {e}")
    stream = None

# ---------------- FACEMESH SETUP ----------------
mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [469, 470, 471, 472, 473]
RIGHT_IRIS = [474, 475, 476, 477, 478]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_UPPER, LEFT_EYE_LOWER = 159, 145
RIGHT_EYE_UPPER, RIGHT_EYE_LOWER = 386, 374
LEFT_BROW = [70, 63, 105]
RIGHT_BROW = [336, 296, 334]
LEFT_EYE_IDX, RIGHT_EYE_IDX = 33, 263
NOSE_IDX, CHIN_IDX = 1, 152

# ---------------- HELPERS ----------------
def get_landmark_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

def compute_eye_openness(pts3):
    try:
        lu, ll = pts3[LEFT_EYE_UPPER][:2], pts3[LEFT_EYE_LOWER][:2]
        ru, rl = pts3[RIGHT_EYE_UPPER][:2], pts3[RIGHT_EYE_LOWER][:2]
        left_outer, left_inner = pts3[LEFT_EYE_CORNERS[0]][:2], pts3[LEFT_EYE_CORNERS[1]][:2]
        right_inner, right_outer = pts3[RIGHT_EYE_CORNERS[0]][:2], pts3[RIGHT_EYE_CORNERS[1]][:2]
    except Exception:
        return 1.0
    left_norm = abs(lu[1]-ll[1]) / (np.linalg.norm(left_outer-left_inner)+1e-8)
    right_norm = abs(ru[1]-rl[1]) / (np.linalg.norm(right_inner-right_outer)+1e-8)
    val = float((left_norm+right_norm)/2)
    t = (val-0.012)/(0.08-0.012)
    return max(0.0, min(1.0, t))

def compute_pupil_normalized(pts3):
    try:
        left_iris = np.mean(pts3[LEFT_IRIS][:, :2], axis=0)
        right_iris = np.mean(pts3[RIGHT_IRIS][:, :2], axis=0)
        left_outer, left_inner = pts3[LEFT_EYE_CORNERS[0]][:2], pts3[LEFT_EYE_CORNERS[1]][:2]
        right_inner, right_outer = pts3[RIGHT_EYE_CORNERS[0]][:2], pts3[RIGHT_EYE_CORNERS[1]][:2]
    except Exception:
        return {"x": 0.0, "y": 0.0}
    left_center = (left_outer + left_inner) / 2.0
    right_center = (right_outer + right_inner) / 2.0
    left_offset = (left_iris - left_center) / (np.linalg.norm(left_outer - left_inner) + 1e-8)
    right_offset = (right_iris - right_center) / (np.linalg.norm(right_outer - right_inner) + 1e-8)
    avg_offset = (left_offset + right_offset) / 2.0
    return {"x": float(avg_offset[0]), "y": float(avg_offset[1])}

def compute_head_rotation(pts3):
    try:
        left, right, nose, chin = pts3[LEFT_EYE_IDX], pts3[RIGHT_EYE_IDX], pts3[NOSE_IDX], pts3[CHIN_IDX]
    except IndexError:
        return {'pitch':0.0,'yaw':0.0,'roll':0.0}
    eye_line = right - left
    nose_chin = chin - nose
    normal = np.cross(eye_line, nose_chin)
    normal /= np.linalg.norm(normal)+1e-8
    roll = math.degrees(math.atan2(eye_line[1], eye_line[0]))
    pitch = math.degrees(math.asin(max(-1,min(1,-normal[1]))))
    yaw = math.degrees(math.asin(max(-1,min(1,normal[0]))))
    return {'pitch':pitch,'yaw':yaw,'roll':roll}

def compute_brow_value(pts3):
    try:
        left_eye_mid = (pts3[LEFT_EYE_CORNERS[0]] + pts3[LEFT_EYE_CORNERS[1]]) / 2.0
        right_eye_mid = (pts3[RIGHT_EYE_CORNERS[0]] + pts3[RIGHT_EYE_CORNERS[1]]) / 2.0
        left_brow_mid = np.mean(pts3[LEFT_BROW], axis=0)
        right_brow_mid = np.mean(pts3[RIGHT_BROW], axis=0)
        avg_dist = ((left_brow_mid[1]-left_eye_mid[1]) + (right_brow_mid[1]-right_eye_mid[1]))/2.0
        eye_w = (np.linalg.norm(pts3[LEFT_EYE_CORNERS[0]][:2]-pts3[LEFT_EYE_CORNERS[1]][:2]) +
                 np.linalg.norm(pts3[RIGHT_EYE_CORNERS[0]][:2]-pts3[RIGHT_EYE_CORNERS[1]][:2]))/2.0
        return float(avg_dist / (eye_w + 1e-8))
    except Exception:
        return 0.0

# ---------------- WRITE JSON ----------------
def write_state():
    with state_lock:
        with open(STATE_FILE, "w") as f:
            json.dump(LATEST_STATE, f)

# ---------------- HTTP SERVER ----------------
class StateHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/state":
            with state_lock:
                data = json.dumps(LATEST_STATE)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return  # disable logging

def run_server():
    server = HTTPServer((HTTP_HOST, HTTP_PORT), StateHandler)
    print(f"HTTP server running at http://{HTTP_HOST}:{HTTP_PORT}/state")
    server.serve_forever()

# ---------------- MAIN LOOP ----------------
threading.Thread(target=run_server, daemon=True).start()

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if result.multi_face_landmarks:
            pts3 = get_landmark_array(result.multi_face_landmarks[0].landmark)
            head = compute_head_rotation(pts3)
            pupil = compute_pupil_normalized(pts3)
            eye = compute_eye_openness(pts3)
            brow = compute_brow_value(pts3)
            with state_lock:
                LATEST_STATE.update({
                    "head_rotation": head,
                    "pupil": pupil,
                    "eye_open": eye,
                    "brows": brow,
                    "volume": volume_level,
                    "speaking": speaking
                })
        write_state()
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
if stream:
    stream.stop()
