import cv2
import mediapipe as mp
import numpy as np
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import sounddevice as sd

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
LEFT_IRIS = [469, 470, 471, 472, 473]
RIGHT_IRIS = [474, 475, 476, 477, 478]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145
RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263
LEFT_BROW = [70, 63, 105]
RIGHT_BROW = [336, 296, 334]
NOSE_IDX = 1
CHIN_IDX = 152

# ---------------- BLENDING SETTINGS ----------------
NEUTRAL_LERP_FACTOR = 0.02
BROW_NEUTRAL_BLEND = 0.005
BROWS_LERP = 0.2

# ---------------- EMOTION RECOGNITION SETTINGS ----------------
TEMPLATES_DIR = r"D:\code\python\PyNG-Tuber\templates"
DEFAULT_TOLERANCE = 0.06
SWITCH_THRESHOLD = 0.005
INSTANT_CONFIDENCE = 0.85

# Neutral pose tracking
neutral_pose = {'pitch': 0.0, 'yaw': 0.0, 'roll': 25.0}
brow_neutral = None

# Emotion state tracking
templates = {}
displayed_emotion = None
candidate_name = None
candidate_since = None

# Global face data
face_data = {
    "head_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "head_rotation_raw": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "pupil": {"x": 0.0, "y": 0.0},
    "eye_open": 1.0,
    "mouth_open": 0.0,
    "brows": 0.0,
    "volume": 0.0,
    "emotion": "neutral",
    "speaking": False,
    "confidence": 1.0
}

# Audio globals
volume_level = 0.0
speaking = False
VOLUME_THRESHOLD_START = 5e-7
VOLUME_THRESHOLD_STOP = 2e-6

def lerp(a, b, f):
    """Linear interpolation"""
    return a + (b - a) * f

def get_landmark_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0.0] for lm in landmarks], dtype=np.float32)

def compute_head_rotation(pts3):
    try:
        left = pts3[LEFT_EYE_IDX]
        right = pts3[RIGHT_EYE_IDX]
        nose = pts3[NOSE_IDX]
        chin = pts3[CHIN_IDX]
    except IndexError:
        return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

    eye_line = right - left
    nose_chin = chin - nose
    normal = np.cross(eye_line, nose_chin)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    roll = math.degrees(math.atan2(eye_line[1], eye_line[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -normal[1]))))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, normal[0]))))
    return {'pitch': float(pitch), 'yaw': float(yaw), 'roll': float(roll)}

def compute_pupil_normalized(pts3):
    try:
        left_iris = np.mean(pts3[LEFT_IRIS][:, :2], axis=0)
        right_iris = np.mean(pts3[RIGHT_IRIS][:, :2], axis=0)
        left_outer, left_inner = pts3[LEFT_EYE_CORNERS[0]][:2], pts3[LEFT_EYE_CORNERS[1]][:2]
        right_inner, right_outer = pts3[RIGHT_EYE_CORNERS[0]][:2], pts3[RIGHT_EYE_CORNERS[1]][:2]
    except Exception:
        return {"x": 0.0, "y": 0.0}

    left_eye_center = (left_outer + left_inner) / 2.0
    left_eye_width = np.linalg.norm(left_outer - left_inner)
    left_offset = (left_iris - left_eye_center) / (left_eye_width + 1e-8)

    right_eye_center = (right_outer + right_inner) / 2.0
    right_eye_width = np.linalg.norm(right_outer - right_inner)
    right_offset = (right_iris - right_eye_center) / (right_eye_width + 1e-8)

    avg_offset = (left_offset + right_offset) / 2.0
    return {"x": float(avg_offset[0]), "y": float(avg_offset[1])}

def compute_eye_openness(pts3):
    try:
        lu = pts3[LEFT_EYE_UPPER][:2]
        ll = pts3[LEFT_EYE_LOWER][:2]
        ru = pts3[RIGHT_EYE_UPPER][:2]
        rl = pts3[RIGHT_EYE_LOWER][:2]
        left_outer = pts3[LEFT_EYE_CORNERS[0]][:2]
        left_inner = pts3[LEFT_EYE_CORNERS[1]][:2]
        right_inner = pts3[RIGHT_EYE_CORNERS[0]][:2]
        right_outer = pts3[RIGHT_EYE_CORNERS[1]][:2]
    except Exception:
        return 1.0

    left_vert = abs(lu[1] - ll[1])
    left_w = np.linalg.norm(left_outer - left_inner)
    right_vert = abs(ru[1] - rl[1])
    right_w = np.linalg.norm(right_inner - right_outer)

    left_norm = left_vert / (left_w + 1e-8)
    right_norm = right_vert / (right_w + 1e-8)
    val = float((left_norm + right_norm) / 2.0)
    MIN_OV = 0.012
    MAX_OV = 0.08
    t = (val - MIN_OV) / (MAX_OV - MIN_OV)
    return max(0.0, min(1.0, t))

def compute_brow_value(pts3):
    global brow_neutral
    try:
        left_eye_mid = (pts3[LEFT_EYE_CORNERS[0]] + pts3[LEFT_EYE_CORNERS[1]])/2.0
        right_eye_mid = (pts3[RIGHT_EYE_CORNERS[0]] + pts3[RIGHT_EYE_CORNERS[1]])/2.0
        left_brow_mid = np.mean(pts3[LEFT_BROW], axis=0)
        right_brow_mid = np.mean(pts3[RIGHT_BROW], axis=0)
        left_dist = left_brow_mid[1] - left_eye_mid[1]
        right_dist = right_brow_mid[1] - right_eye_mid[1]
        avg_dist = (left_dist + right_dist)/2.0
        left_w = np.linalg.norm(pts3[LEFT_EYE_CORNERS[0]][:2] - pts3[LEFT_EYE_CORNERS[1]][:2])
        right_w = np.linalg.norm(pts3[RIGHT_EYE_CORNERS[0]][:2] - pts3[RIGHT_EYE_CORNERS[1]][:2])
        ref_w = (left_w + right_w) / 2.0 + 1e-8
        avg_dist_norm = avg_dist / ref_w
        
        if brow_neutral is None:
            brow_neutral = avg_dist_norm
        
        brow_neutral = lerp(brow_neutral, avg_dist_norm, BROW_NEUTRAL_BLEND)
        delta = avg_dist_norm - brow_neutral
        return float(delta)
    except Exception:
        return 0.0

def compute_mouth_openness(pts3):
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    try:
        top = pts3[MOUTH_TOP][:2]
        bottom = pts3[MOUTH_BOTTOM][:2]
        mouth_height = abs(top[1] - bottom[1])
        left_eye = pts3[LEFT_EYE_IDX][:2]
        right_eye = pts3[RIGHT_EYE_IDX][:2]
        eye_dist = np.linalg.norm(right_eye - left_eye)
        normalized = mouth_height / (eye_dist + 1e-8)
        return float(min(1.0, normalized * 10.0))
    except Exception:
        return 0.0

def apply_neutral_correction(raw_head_rot):
    """Apply Blender-style neutral pose correction"""
    global neutral_pose
    
    for key in neutral_pose:
        neutral_pose[key] = lerp(neutral_pose[key], raw_head_rot[key], NEUTRAL_LERP_FACTOR)
    
    corrected = {
        'pitch': raw_head_rot['pitch'] - neutral_pose['pitch'],
        'yaw': raw_head_rot['yaw'] - neutral_pose['yaw'],
        'roll': raw_head_rot['roll'] - neutral_pose['roll']
    }
    return corrected

# ---------------- EMOTION RECOGNITION FUNCTIONS ----------------

def euler_to_matrix_3d(p_rad, y_rad, r_rad):
    cp, sp = math.cos(p_rad), math.sin(p_rad)
    cy, sy = math.cos(y_rad), math.sin(y_rad)
    cr, sr = math.cos(r_rad), math.sin(r_rad)

    Rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])

    R = Rz @ Ry @ Rx
    return R

def invert_rotation_and_apply(points, pitch, yaw, roll):
    """Apply inverse rotation to counteract head rotation"""
    p_rad = math.radians(-pitch)
    y_rad = math.radians(-yaw)
    r_rad = math.radians(-roll)
    R_inv = euler_to_matrix_3d(p_rad, y_rad, r_rad)
    
    center = points.mean(axis=0)
    pts_centered = points - center
    pts_rotated = (R_inv @ pts_centered.T).T
    pts_corrected = pts_rotated + center
    return pts_corrected

def normalize_landmarks_no_rotate(pts):
    """Normalize landmarks for emotion matching"""
    if pts.shape[0] == 0:
        return None
    center = pts.mean(axis=0)
    pts_c = pts - center
    if pts.shape[0] > max(LEFT_EYE_IDX, RIGHT_EYE_IDX):
        left = pts[LEFT_EYE_IDX][:2]
        right = pts[RIGHT_EYE_IDX][:2]
        iod = np.linalg.norm(right - left)
    else:
        iod = np.max(np.linalg.norm(pts_c[:, :2], axis=1))
    if iod <= 1e-6:
        iod = 1.0
    pts_n = pts_c[:, :2] / iod
    return pts_n.flatten().tolist()

def orthogonal_procrustes_mae(X, Y):
    """Calculate mean absolute error after procrustes alignment"""
    if X.size == 0 or Y.size == 0:
        return float('inf')
    n = min(X.shape[0], Y.shape[0])
    if n == 0:
        return float('inf')
    Xa = X[:n].copy()
    Ya = Y[:n].copy()
    A = Xa.T @ Ya
    try:
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
    except:
        R = np.eye(2, dtype=np.float32)
    Xr = Xa @ R
    return float(np.mean(np.abs(Xr - Ya)))

def load_templates():
    """Load emotion templates from JSON files"""
    templates = {}
    if not os.path.exists(TEMPLATES_DIR):
        print(f"[WARNING] Templates directory not found: {TEMPLATES_DIR}")
        return templates
    
    for fn in os.listdir(TEMPLATES_DIR):
        if fn.lower().endswith('.json'):
            path = os.path.join(TEMPLATES_DIR, fn)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                samples = [np.array(s, dtype=np.float32) for s in data.get('samples', [])]
                templates[data['name']] = {
                    "samples": samples,
                    "landmark_count": int(data.get('landmark_count', 0)),
                    "tolerance": float(data.get('tolerance', DEFAULT_TOLERANCE)),
                    "path": path
                }
            except Exception as e:
                print(f'[ERROR] Failed to load template {path}: {e}')
    return templates

def match_emotion(pts3, head_rot):
    """Match current face landmarks to emotion templates"""
    global templates
    
    if not templates:
        return None, None, None
    
    # Apply inverse rotation to normalize head pose
    pts3_corrected = invert_rotation_and_apply(
        pts3,
        head_rot['pitch'],
        head_rot['yaw'],
        head_rot['roll']
    )
    
    # Normalize landmarks
    vec = normalize_landmarks_no_rotate(pts3_corrected)
    if vec is None:
        return None, None, None
    
    live_vector = np.array(vec, dtype=np.float32)
    live_pts = live_vector.reshape(-1, 2)
    
    # Find best matching template
    best = None
    for tname, t in templates.items():
        for samp in t['samples']:
            samp_pts = samp.reshape(-1, 2)
            mae = orthogonal_procrustes_mae(live_pts, samp_pts)
            if best is None or mae < best[1]:
                best = (tname, mae, t.get('tolerance', DEFAULT_TOLERANCE))
    
    if best is not None:
        bname, bmae, btol = best
        score = max(0.0, 1.0 - (bmae / btol)) if btol > 0 else 0.0
        return bname, bmae, score
    
    return None, None, None

def update_emotion_state(match_name, match_score):
    """Update emotion state with hysteresis"""
    global displayed_emotion, candidate_name, candidate_since
    
    now = time.monotonic()
    
    if match_name != candidate_name:
        candidate_name = match_name
        candidate_since = now
    else:
        if candidate_name is not None:
            # Instant switch if high confidence
            if match_score is not None and match_score >= INSTANT_CONFIDENCE:
                if displayed_emotion != candidate_name:
                    displayed_emotion = candidate_name
                    candidate_since = now
            # Gradual switch with threshold
            elif displayed_emotion != candidate_name and (now - candidate_since) >= SWITCH_THRESHOLD:
                displayed_emotion = candidate_name
        else:
            # Return to neutral
            if displayed_emotion is not None and (now - candidate_since) >= SWITCH_THRESHOLD:
                displayed_emotion = None

# ---------------- AUDIO CALLBACK ----------------

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

# ---------------- HTTP SERVER ----------------

class FaceTrackingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        body = json.dumps(face_data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

def run_http_server():
    server = HTTPServer(("127.0.0.1", 80), FaceTrackingHandler)
    print("HTTP server started on http://127.0.0.1:80")
    print("Neutral pose auto-calibration enabled!")
    print("Emotion recognition enabled!")
    print("Streaming BOTH corrected + raw head rotation!")
    server.serve_forever()

# ---------------- MAIN LOOP ----------------

def main():
    global face_data, templates, displayed_emotion
    
    # Load templates
    templates = load_templates()
    print(f"Loaded {len(templates)} emotion templates: {list(templates.keys())}")
    
    # Start HTTP server
    server_thread = threading.Thread(target=run_http_server, daemon=True)
    server_thread.start()
    
    # Start audio stream
    stream = sd.InputStream(callback=audio_callback)
    stream.start()
    
    cap = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        print("Face tracking started. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                pts3 = get_landmark_array(landmarks)
                
                # Raw computations
                raw_head_rot = compute_head_rotation(pts3)
                pupil_pos = compute_pupil_normalized(pts3)
                eye_open = compute_eye_openness(pts3)
                mouth_open = compute_mouth_openness(pts3)
                brows_raw = compute_brow_value(pts3)
                
                # Emotion recognition
                match_name, match_mae, match_score = match_emotion(pts3, raw_head_rot)
                update_emotion_state(match_name, match_score)
                
                # Apply neutral pose correction (for body rigging)
                corrected_head_rot = apply_neutral_correction(raw_head_rot)
                
                # Update face_data
                face_data["head_rotation"] = corrected_head_rot
                face_data["head_rotation_raw"] = raw_head_rot
                face_data["pupil"] = pupil_pos
                face_data["eye_open"] = eye_open
                face_data["mouth_open"] = mouth_open
                face_data["brows"] = brows_raw
                face_data["volume"] = volume_level
                face_data["speaking"] = speaking
                face_data["emotion"] = displayed_emotion or "neutral"
                face_data["confidence"] = match_score if match_score is not None else 1.0
            
            # Display current emotion on frame
            emotion_text = f"Emotion: {displayed_emotion or 'neutral'}"
            cv2.putText(frame, emotion_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        stream.stop()

if __name__ == "__main__":
    main()