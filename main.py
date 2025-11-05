import cv2
import mediapipe as mp
import numpy as np
import os, json, time
from PIL import Image
import sounddevice as sd
import socket
import threading
import math


# ----------------- config / defaults -----------------
TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# iris indices according to MediaPipe Face Mesh
LEFT_IRIS = [469, 470, 471, 472, 473]
RIGHT_IRIS = [474, 475, 476, 477, 478]


# eye corner indices for normalization (left / right)
LEFT_EYE_CORNERS = (33, 133)   # left eye: outer, inner
RIGHT_EYE_CORNERS = (362, 263) # right eye: inner, outer
# eye indices for eye openness and such
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145
RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374
# Eyebrow indices
LEFT_BROW = [70, 63, 105] # left inner/mid/out brow
RIGHT_BROW = [336, 296, 334] # right inner/mid/out brow

NOSE_IDX = 1
CHIN_IDX = 152  # approximate chin tip in MediaPipe Face Mesh


MOUTH_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95, 78
]

# Inner mouth contour (the opening — upper + lower inner lip)
MOUTH_INNER = [
    191, 80, 81, 82, 13, 312, 311,
    310, 415, 308
]


tneutral_brow = None

DEFAULT_TOLERANCE = 0.06

# runtime thresholds (can be reloaded from settings.json via 'l')
SWITCH_THRESHOLD = 0.005
INSTANT_CONFIDENCE = 0.85

SPEAK_SWITCH_THRESHOLD = 0.05
VOLUME_THRESHOLD_START = 5e-7
VOLUME_THRESHOLD_STOP = 2e-6

# textures dir and map
character_textures_dir = "textures/placeholders"
textures = {
    "neutral":  {"main": f"{character_textures_dir}/neutral.png",  "speaking": f"{character_textures_dir}/neutral_speaking.png"},
    "happy":    {"main": f"{character_textures_dir}/happy.png",    "speaking": f"{character_textures_dir}/happy_speaking.png"},
    "angry":    {"main": f"{character_textures_dir}/angry.png",    "speaking": f"{character_textures_dir}/angry_speaking.png"},
    "surprised":{"main": f"{character_textures_dir}/surprise.png","speaking": f"{character_textures_dir}/surprise_speaking.png"},
    "mewing":   {"main": f"{character_textures_dir}/mewing.png",   "speaking": f"{character_textures_dir}/mewing_speaking.png"},
    "peculiar": {"main": f"{character_textures_dir}/peculiar.png", "speaking": f"{character_textures_dir}/peculiar_speaking.png"}
}

# networking (stream to Blender)
STREAM_HOST = '127.0.0.1'
STREAM_PORT = 5005

DATA_STREAMING_MODE = False

# ----------------- networking (TCP server) -----------------
clients = []
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((STREAM_HOST, STREAM_PORT))
server_socket.listen(1)

def accept_clients_loop():
    while True:
        try:
            conn, addr = server_socket.accept()
            print(f"[stream] client connected: {addr}")
            clients.append(conn)
            threading.Thread(target=client_monitor, args=(conn,addr), daemon=True).start()
        except Exception as e:
            print(f"[stream] accept failed: {e}")
            break

def client_monitor(conn, addr):
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
    finally:
        if conn in clients: clients.remove(conn)
        conn.close()
        print(f"[stream] client disconnected: {addr}")

threading.Thread(target=accept_clients_loop, daemon=True).start()

def broadcast_state(state_dict):
    data = (json.dumps(state_dict) + "\n").encode('utf-8')
    dead = []
    for c in list(clients):
        try:
            c.sendall(data)
        except Exception:
            dead.append(c)
    for d in dead:
        if d in clients: clients.remove(d)

# ----------------- audio (volume + VAD with hysteresis) -----------------
volume_level = 0.0
speaking = False

def audio_callback(indata, frames, time_, status):
    global volume_level, speaking
    # RMS-ish normalized by frames
    volume_norm = float(np.linalg.norm(indata) / frames)
    volume_level = volume_norm
    # hysteresis for speaking detection
    if speaking:
        if volume_norm < VOLUME_THRESHOLD_STOP:
            speaking = False
    else:
        if volume_norm > VOLUME_THRESHOLD_START:
            speaking = True

stream = sd.InputStream(callback=audio_callback)
stream.start()

# ----------------- helpers -----------------

def get_eye_center(pts, indices):
    selected = pts[indices]
    return np.mean(selected, axis=0)

def compute_brow_value(pts3):
    global tneutral_brow
    try:
        left_eye_mid = (pts3[LEFT_EYE_CORNERS[0]] + pts3[LEFT_EYE_CORNERS[1]])/2.0
        right_eye_mid = (pts3[RIGHT_EYE_CORNERS[0]] + pts3[RIGHT_EYE_CORNERS[1]])/2.0

        left_brow_mid = np.mean(pts3[LEFT_BROW], axis=0)
        right_brow_mid = np.mean(pts3[RIGHT_BROW], axis=0)

        # vertical distances
        left_dist = left_brow_mid[1] - left_eye_mid[1]
        right_dist = right_brow_mid[1] - right_eye_mid[1]
        avg_dist = (left_dist + right_dist)/2.0

        # normalize by average eye width
        left_w = np.linalg.norm(pts3[LEFT_EYE_CORNERS[0]][:2] - pts3[LEFT_EYE_CORNERS[1]][:2])
        right_w = np.linalg.norm(pts3[RIGHT_EYE_CORNERS[0]][:2] - pts3[RIGHT_EYE_CORNERS[1]][:2])
        ref_w = (left_w + right_w) / 2.0 + 1e-8

        avg_dist_norm = avg_dist / ref_w

        # initialize neutral reference
        if tneutral_brow is None:
            tneutral_brow = avg_dist_norm

        delta = avg_dist_norm - tneutral_brow
        return float(delta)
    except Exception:
        return 0.0

def compute_eye_openness(pts3):
    """Return a normalized eye openness value in [0,1]."""
    try:
        lu = pts3[LEFT_EYE_UPPER][:2]
        ll = pts3[LEFT_EYE_LOWER][:2]
        ru = pts3[RIGHT_EYE_UPPER][:2]
        rl = pts3[RIGHT_EYE_LOWER][:2]
        # eye widths for normalization
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
    t = max(0.0, min(1.0, t))
    return t

def compute_pupil_normalized(pts3):
    try:
        left_iris = np.mean(pts3[LEFT_IRIS][:, :2], axis=0)
        right_iris = np.mean(pts3[RIGHT_IRIS][:, :2], axis=0)
        left_outer, left_inner = pts3[LEFT_EYE_CORNERS[0]][:2], pts3[LEFT_EYE_CORNERS[1]][:2]
        right_inner, right_outer = pts3[RIGHT_EYE_CORNERS[0]][:2], pts3[RIGHT_EYE_CORNERS[1]][:2]
    except Exception:
        return {"x":0.0, "y":0.0}

    left_eye_center = (left_outer + left_inner) / 2.0
    left_eye_width = np.linalg.norm(left_outer - left_inner)
    left_offset = (left_iris - left_eye_center) / (left_eye_width + 1e-8)

    right_eye_center = (right_outer + right_inner) / 2.0
    right_eye_width = np.linalg.norm(right_outer - right_inner)
    right_offset = (right_iris - right_eye_center) / (right_eye_width + 1e-8)

    avg_offset = (left_offset + right_offset) / 2.0

    return {"x": float(avg_offset[0]), "y": float(avg_offset[1])}

def load_image_force(path):
    img = Image.open(path)
    img.load()
    return cv2.cvtColor(np.array(img.convert('RGBA').copy()), cv2.COLOR_RGBA2BGRA)

def load_textures():
    loaded = {}
    for e, states in textures.items():
        loaded[e] = {}
        for s, p in states.items():
            if os.path.exists(p):
                try:
                    loaded[e][s] = load_image_force(p)
                except Exception as ex:
                    print(f"[load_textures] failed to read {p}: {ex}")
    return loaded

def load_settings(cfg_path='settings.json'):
    global SWITCH_THRESHOLD, SPEAK_SWITCH_THRESHOLD, VOLUME_THRESHOLD_START, VOLUME_THRESHOLD_STOP, INSTANT_CONFIDENCE, DATA_STREAMING_MODE
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            SWITCH_THRESHOLD = float(cfg.get('switch_threshold', SWITCH_THRESHOLD))
            SPEAK_SWITCH_THRESHOLD = float(cfg.get('speak_switch_threshold', SPEAK_SWITCH_THRESHOLD))
            VOLUME_THRESHOLD_START = float(cfg.get('volume_threshold_start', VOLUME_THRESHOLD_START))
            VOLUME_THRESHOLD_STOP = float(cfg.get('volume_threshold_stop', VOLUME_THRESHOLD_STOP))
            INSTANT_CONFIDENCE = float(cfg.get('instant_confidence', INSTANT_CONFIDENCE))
            DATA_STREAMING_MODE = bool(cfg.get("data_streaming_mode", DATA_STREAMING_MODE))
            print(f"[load_settings] switch={SWITCH_THRESHOLD}, speak_switch={SPEAK_SWITCH_THRESHOLD}, vol_start={VOLUME_THRESHOLD_START}, vol_stop={VOLUME_THRESHOLD_STOP}, instant_conf={INSTANT_CONFIDENCE}")
        except Exception as e:
            print('[load_settings] failed:', e)

# template / matching helpers (unchanged)

def save_template(name, samples, tolerance=DEFAULT_TOLERANCE):
    data = {"name": name, "samples": samples, "landmark_count": int(len(samples[0])//2) if samples else 0, "tolerance": float(tolerance), "timestamp": time.time()}
    path = os.path.join(TEMPLATES_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[saved] template '{name}' -> {path}")

def load_templates():
    templates = {}
    for fn in os.listdir(TEMPLATES_DIR):
        if fn.lower().endswith('.json'):
            path = os.path.join(TEMPLATES_DIR, fn)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                samples = [np.array(s, dtype=np.float32) for s in data.get('samples', [])]
                templates[data['name']] = {"samples": samples, "landmark_count": int(data.get('landmark_count',0)), "tolerance": float(data.get('tolerance', DEFAULT_TOLERANCE)), "path": path}
            except Exception as e:
                print('Failed to load template', path, e)
    return templates

def get_landmark_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0.0] for lm in landmarks], dtype=np.float32)

def normalize_landmarks_no_rotate(pts):
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
    if X.size == 0 or Y.size == 0: return float('inf')
    n = min(X.shape[0], Y.shape[0])
    if n == 0: return float('inf')
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

# ----------------- head rotation using 3 points (improved) -----------------

def compute_head_rotation(pts3):
    try:
        left = pts3[LEFT_EYE_IDX]
        right = pts3[RIGHT_EYE_IDX]
        nose = pts3[NOSE_IDX]
        chin = pts3[CHIN_IDX]
    except IndexError:
        return {'pitch':0.0, 'yaw':0.0, 'roll':0.0}

    eye_line = right - left
    nose_chin = chin - nose
    normal = np.cross(eye_line, nose_chin)
    normal = normal / (np.linalg.norm(normal)+1e-8)

    # roll: eye line angle
    roll = math.degrees(math.atan2(eye_line[1], eye_line[0]))
    # pitch: asin of -normal.y
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -normal[1]))))
    # yaw: asin of normal.x
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, normal[0]))))

    return {'pitch': float(pitch), 'yaw': float(yaw), 'roll': float(roll)}

# --------------- new: inverse rotation counteracting head rotate -------------------
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
    # points: Nx3 numpy array
    # Rotation to counteract (negative angles)
    p_rad = math.radians(-pitch)
    y_rad = math.radians(-yaw)
    r_rad = math.radians(-roll)
    R_inv = euler_to_matrix_3d(p_rad, y_rad, r_rad)  # inverse rotation matrix

    # Center point as mean of points
    center = points.mean(axis=0)

    # Translate points to origin, rotate, then translate back
    pts_centered = points - center
    pts_rotated = (R_inv @ pts_centered.T).T
    pts_corrected = pts_rotated + center
    return pts_corrected
# ----------------- startup loads -----------------
loaded_textures = load_textures()
load_settings()
templates = load_templates()
print('Loaded templates:', list(templates.keys()))
print('Loaded textures:', list(loaded_textures.keys()))

# GUI/window
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
GREEN_KEY = (0,255,0,255)
if not DATA_STREAMING_MODE:
    cv2.namedWindow('PNG Tuber', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('PNG Tuber', WINDOW_WIDTH, WINDOW_HEIGHT)

# runtime state
cap = cv2.VideoCapture(0)
recording = False
record_samples = []
record_interval = 0.0
last_record_time = 0.0
candidate_name = None
candidate_since = None
displayed_emotion = None
speaking_state = False
speaking_since = None

# ----------------- main loop -----------------
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            # runtime variables
            live_vector = None
            match_name = None
            match_mae = None
            match_score = None
            head_rot = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
            pupil = {'x': 0.0, 'y': 0.0}

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pts3 = get_landmark_array(lm)


                # compute head rotation first
                head_rot = compute_head_rotation(pts3)  # Use raw landmarks to get actual face rotation

                # apply inverse rotation to counteract head rotation on pts3
                pts3 = invert_rotation_and_apply(
                    pts3,
                    head_rot['pitch'],
                    head_rot['yaw'],
                    head_rot['roll']
                )

                # now perform pupil / iris tracking and feature extraction on corrected landmarks

                pupil = compute_pupil_normalized(pts3)

                vec = normalize_landmarks_no_rotate(pts3)
                if vec is not None:
                    live_vector = np.array(vec, dtype=np.float32)
                    live_pts = live_vector.reshape(-1, 2)

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
                        match_name, match_mae, match_score = bname, bmae, score


                if not DATA_STREAMING_MODE:
                    for face_landmarks in results.multi_face_landmarks:
                        mouth_indices = MOUTH_OUTER + MOUTH_INNER
                        for idx in mouth_indices:
                            x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                            y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # small green dots


            now = time.monotonic()

            # emotion switching
            if match_name != candidate_name:
                candidate_name = match_name
                candidate_since = now
            else:
                if candidate_name is not None:
                    if match_score is not None and match_score >= INSTANT_CONFIDENCE:
                        if displayed_emotion != candidate_name:
                            displayed_emotion = candidate_name
                            candidate_since = now
                    elif displayed_emotion != candidate_name and (now - candidate_since) >= SWITCH_THRESHOLD:
                        displayed_emotion = candidate_name
                else:
                    if displayed_emotion is not None and (now - candidate_since) >= SWITCH_THRESHOLD:
                        displayed_emotion = None

            # speaking state switching
            if speaking != speaking_state:
                if speaking_since is None:
                    speaking_since = now
                elif (now - speaking_since) >= SPEAK_SWITCH_THRESHOLD:
                    speaking_state = speaking
                    speaking_since = None
            else:
                speaking_since = None

            mouth_data = None
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pts3 = get_landmark_array(lm)

                # compute head rotation for mouth payload (using raw pts again to represent actual head rotation)
                head_rot = compute_head_rotation(pts3)

                # apply inverse rotation on pts3 for mouth landmarks stability
                pts3 = invert_rotation_and_apply(
                    pts3,
                    head_rot['pitch'],
                    head_rot['yaw'],
                    head_rot['roll']
                )

                pupil = compute_pupil_normalized(pts3)
                eye_open = compute_eye_openness(pts3)
                brow_val = compute_brow_value(pts3)

                # --- new: compute face bbox (normalized image coords) ---
                xs = pts3[:,0]
                ys = pts3[:,1]
                minx, maxx = float(xs.min()), float(xs.max())
                miny, maxy = float(ys.min()), float(ys.max())

                # --- new: gather mouth landmarks (image-normalized coords) ---
                mouth_idxs = MOUTH_OUTER + MOUTH_INNER
                mouth_landmarks = []
                for idx in mouth_idxs:
                    if idx < len(pts3):
                        x = float(pts3[idx][0])
                        y = float(pts3[idx][1])
                        mouth_landmarks.append({"i": int(idx), "x": x, "y": y})
                outer_lm = []
                inner_lm = []
                for idx in MOUTH_OUTER:
                    if idx < len(pts3):
                        outer_lm.append({"i": int(idx), "x": float(pts3[idx][0]), "y": float(pts3[idx][1])})
                for idx in MOUTH_INNER:
                    if idx < len(pts3):
                        inner_lm.append({"i": int(idx), "x": float(pts3[idx][0]), "y": float(pts3[idx][1])})
                mouth_data = {
                    "bbox": [minx, miny, maxx, maxy],
                    "outer": outer_lm,
                    "inner": inner_lm
                }

                # pack state
                state = {
                    "head_rotation": head_rot,
                    "pupil": pupil,
                    "eye_open": eye_open,
                    "brows": brow_val,
                    "volume": volume_level,
                    "emotion": displayed_emotion or "neutral",
                    "speaking": speaking,
                    "confidence": 1.0,
                    "mouth": mouth_data  # <-- new: mouth payload
                }

                # send to Blender
                broadcast_state(state)

            # show preview if not streaming
            if not DATA_STREAMING_MODE:
                cv2.imshow("PNG Tuber", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break
            # --- Rendering --- (unchanged)
            if DATA_STREAMING_MODE:
                cv2.imshow("Data Streaming Webcam", frame)
            else:
                y = 20
                cv2.putText(frame, f"Templates: {', '.join(templates.keys()) or 'none'}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y += 22

                #tuber_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 4), dtype=np.uint8)
                #tuber_frame[:, :] = GREEN_KEY
                #emotion = displayed_emotion or 'neutral'
                #state_tex = 'speaking' if speaking_state and emotion in loaded_textures and 'speaking' in loaded_textures[emotion] else 'main'
                #texture = loaded_textures.get(emotion, {}).get(state_tex)
                #if texture is not None:
                #    th, tw = texture.shape[:2]
                #    scale = min(WINDOW_WIDTH / tw, WINDOW_HEIGHT / th)
                #    new_w, new_h = max(1, int(tw * scale)), max(1, int(th * scale))
                #    resized = cv2.resize(texture, (new_w, new_h), interpolation=cv2.INTER_AREA)
                #    x = (WINDOW_WIDTH - new_w) // 2
                #    y = (WINDOW_HEIGHT - new_h) // 2
                #    y1, y2 = y, y + new_h
                #    x1, x2 = x, x + new_w
                #    alpha_s = resized[:, :, 3:4] / 255.0
                #    alpha_l = 1.0 - alpha_s
                #    tuber_frame[y1:y2, x1:x2, :3] = (
                #        alpha_s * resized[:, :, :3] + alpha_l * tuber_frame[y1:y2, x1:x2, :3]
                #    )
                #cv2.imshow('PNG Tuber', tuber_frame)

            # key handling (unchanged)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                recording = not recording
                if recording:
                    record_samples = []
                    interval_input = input('Recording interval in seconds (e.g. 0.5): ').strip()
                    try:
                        record_interval = float(interval_input)
                    except:
                        record_interval = 0.0
                    last_record_time = 0.0
                    print(f"[record] started with interval {record_interval}s. Press 'r' to stop.")
                else:
                    if not record_samples:
                        print('[record] stopped but no samples recorded.')
                    else:
                        name = input('Template name (single word): ').strip()
                        if not name:
                            print('Aborted save.')
                        else:
                            tol_input = input(f'Tolerance MAE [default {DEFAULT_TOLERANCE}]:').strip()
                            try:
                                tol = float(tol_input) if tol_input else DEFAULT_TOLERANCE
                            except:
                                tol = DEFAULT_TOLERANCE
                            path = os.path.join(TEMPLATES_DIR, f"{name}.json")
                            if os.path.exists(path):
                                overwrite = input(f"Template '{name}' exists. Overwrite? (y/N): ").strip().lower()
                                if overwrite != 'y':
                                    print('Aborted save.')
                                else:
                                    save_template(name, record_samples, tol)
                                    templates = load_templates()
                            else:
                                save_template(name, record_samples, tol)
                                templates = load_templates()
                    record_samples = []

    except Exception as e:
        print("[error]", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        stream.stop()
        server_socket.close()