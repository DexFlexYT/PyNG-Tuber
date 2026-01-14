import cv2
import mediapipe as mp
import numpy as np
import json
import math
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
NEUTRAL_LERP_FACTOR = 0.005      # How fast neutral pose adapts
BROW_NEUTRAL_BLEND = 0.005       # How fast brow neutral adapts
BROWS_LERP = 0.2                 # Brow movement smoothing

# Neutral pose tracking
neutral_pose = {'pitch': 0.0, 'yaw': 0.0, 'roll': 25.0}
brow_neutral = None

# Global face data (streams BOTH corrected AND raw data)
face_data = {
    "head_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},     # CORRECTED for body
    "head_rotation_raw": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, # RAW for pupils
    "pupil": {"x": 0.0, "y": 0.0},
    "eye_open": 1.0,
    "mouth_open": 0.0,
    "brows": 0.0,         # Already neutral-corrected delta
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

# HTTP Server Handler
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
    print("Streaming BOTH corrected + raw head rotation!")
    server.serve_forever()

def main():
    global face_data
    
    server_thread = threading.Thread(target=run_http_server, daemon=True)
    server_thread.start()
    
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
                
                # Apply neutral pose correction (for body rigging)
                corrected_head_rot = apply_neutral_correction(raw_head_rot)
                
                # Update face_data with BOTH values
                face_data["head_rotation"] = corrected_head_rot      # Body rigging (corrected)
                face_data["head_rotation_raw"] = raw_head_rot        # Pupils (raw, no drift)
                face_data["pupil"] = pupil_pos
                face_data["eye_open"] = eye_open
                face_data["mouth_open"] = mouth_open
                face_data["brows"] = brows_raw
                face_data["volume"] = volume_level
                face_data["speaking"] = speaking
                face_data["emotion"] = "neutral"
                face_data["confidence"] = 1.0
            
            cv2.imshow('Face Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        stream.stop()

if __name__ == "__main__":
    main()
