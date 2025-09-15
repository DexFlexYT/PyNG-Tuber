import cv2
import mediapipe as mp
import numpy as np
import os, json, time
from PIL import Image
import sounddevice as sd

# ----------------- config / defaults -----------------
TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263
DEFAULT_TOLERANCE = 0.06

# runtime thresholds (can be reloaded from settings.json via 'l')
SWITCH_THRESHOLD = 0.005  
INSTANT_CONFIDENCE = 0.85      # if score >= this, switch instantly

SPEAK_SWITCH_THRESHOLD = 0.0    # speak debounce (s)
VOLUME_THRESHOLD_START = 5e-7   # low -> easy to trigger start
VOLUME_THRESHOLD_STOP  = 2e-6   # high -> need stronger silence to stop


# textures dir and map
character_textures_dir = "textures/dex"
textures = {
    "neutral":  {"main": f"{character_textures_dir}/neutral.png",  "speaking": f"{character_textures_dir}/neutral_speaking.png"},
    "happy":    {"main": f"{character_textures_dir}/happy.png",    "speaking": f"{character_textures_dir}/happy_speaking.png"},
    "angry":    {"main": f"{character_textures_dir}/angry.png",    "speaking": f"{character_textures_dir}/angry_speaking.png"},
    "surprised":{"main": f"{character_textures_dir}/surprise.png","speaking": f"{character_textures_dir}/surprise_speaking.png"},
    "mewing":   {"main": f"{character_textures_dir}/mewing.png",   "speaking": f"{character_textures_dir}/mewing_speaking.png"},
    "peculiar": {"main": f"{character_textures_dir}/peculiar.png", "speaking": f"{character_textures_dir}/peculiar_speaking.png"}
}

# ----------------- helpers -----------------
def load_image_force(path):
    """Load PNG right from disk, force read to avoid caching issues."""
    img = Image.open(path)
    img.load()                 # force disk read
    return cv2.cvtColor(np.array(img.convert("RGBA").copy()), cv2.COLOR_RGBA2BGRA)

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

def load_settings(cfg_path="settings.json"):
    global SWITCH_THRESHOLD, SPEAK_SWITCH_THRESHOLD, VOLUME_THRESHOLD_START, VOLUME_THRESHOLD_STOP, INSTANT_CONFIDENCE
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            SWITCH_THRESHOLD = float(cfg.get("switch_threshold", SWITCH_THRESHOLD))
            SPEAK_SWITCH_THRESHOLD = float(cfg.get("speak_switch_threshold", SPEAK_SWITCH_THRESHOLD))
            VOLUME_THRESHOLD_START = float(cfg.get("volume_threshold_start", VOLUME_THRESHOLD_START))
            VOLUME_THRESHOLD_STOP = float(cfg.get("volume_threshold_stop", VOLUME_THRESHOLD_STOP))
            INSTANT_CONFIDENCE = float(cfg.get("instant_confidence", INSTANT_CONFIDENCE))
            print(f"[load_settings] switch={SWITCH_THRESHOLD}, speak_switch={SPEAK_SWITCH_THRESHOLD}, volume_threshold_start={VOLUME_THRESHOLD_START}, volume_threshold_stop={VOLUME_THRESHOLD_STOP}, instant_conf={INSTANT_CONFIDENCE}")
        except Exception as e:
            print("[load_settings] failed:", e)
    else:
        # no file: keep defaults
        pass

def save_template(name, samples, tolerance=DEFAULT_TOLERANCE):
    data = {"name": name, "samples": samples, "landmark_count": int(len(samples[0])//2) if samples else 0, "tolerance": float(tolerance), "timestamp": time.time()}
    path = os.path.join(TEMPLATES_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] template '{name}' -> {path}")

def load_templates():
    templates = {}
    for fn in os.listdir(TEMPLATES_DIR):
        if fn.lower().endswith(".json"):
            path = os.path.join(TEMPLATES_DIR, fn)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                samples = [np.array(s, dtype=np.float32) for s in data.get("samples", [])]
                templates[data["name"]] = {"samples": samples, "landmark_count": int(data.get("landmark_count", 0)), "tolerance": float(data.get("tolerance", DEFAULT_TOLERANCE)), "path": path}
            except Exception as e:
                print("Failed to load template", path, e)
    return templates

def get_landmark_array(landmarks):
    return np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)

def normalize_landmarks_no_rotate(pts):
    if pts.shape[0] == 0:
        return None
    center = pts.mean(axis=0)
    pts_c = pts - center
    if pts.shape[0] > max(LEFT_EYE_IDX, RIGHT_EYE_IDX):
        left = pts[LEFT_EYE_IDX]
        right = pts[RIGHT_EYE_IDX]
        iod = np.linalg.norm(right - left)
    else:
        iod = np.max(np.linalg.norm(pts_c, axis=1))
    if iod <= 1e-6:
        iod = 1.0
    pts_n = pts_c / iod
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

# ----------------- audio (simple VAD) -----------------
speaking = False

def audio_callback(indata, frames, time_, status):
    global speaking
    volume_norm = np.linalg.norm(indata) / frames

    if speaking:
        # already talking -> require stronger silence to stop
        if volume_norm < VOLUME_THRESHOLD_STOP:
            speaking = False
    else:
        # currently silent -> easier to trigger speech
        if volume_norm > VOLUME_THRESHOLD_START:
            speaking = True
# start audio stream
stream = sd.InputStream(callback=audio_callback)
stream.start()

# ----------------- startup loads -----------------
loaded_textures = load_textures()
load_settings()
templates = load_templates()
print("Loaded templates:", list(templates.keys()))
print("Loaded textures:", list(loaded_textures.keys()))

# GUI/window
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
GREEN_KEY = (0, 255, 0, 255)
cv2.namedWindow("PNG Tuber", cv2.WINDOW_NORMAL)
cv2.resizeWindow("PNG Tuber", WINDOW_WIDTH, WINDOW_HEIGHT)

# ----------------- runtime state -----------------
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

# ----------------- reload helper -----------------
def reload_settings_and_textures():
    global templates, loaded_textures
    print("[reload] triggered")
    load_settings()
    templates = load_templates()
    loaded_textures.clear()
    # force reload from disk
    for emotion, tex in textures.items():
        loaded_textures[emotion] = {}
        for state, path in tex.items():
            if os.path.exists(path):
                try:
                    loaded_textures[emotion][state] = load_image_force(path)
                    print(f"[reload] loaded {path}")
                except Exception as e:
                    print(f"[reload] failed to load {path}: {e}")
    print("[reload] done")

# ----------------- main loop -----------------
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            live_vector = None
            match_name = None
            match_mae = None
            match_score = None

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pts = get_landmark_array(lm)
                vec = normalize_landmarks_no_rotate(pts)
                if vec is not None:
                    live_vector = np.array(vec, dtype=np.float32)
                    best = None
                    live_pts = live_vector.reshape(-1, 2)
                    # find best template match
                    for tname, t in templates.items():
                        for samp in t["samples"]:
                            samp_pts = samp.reshape(-1, 2)
                            mae = orthogonal_procrustes_mae(live_pts, samp_pts)
                            if best is None or mae < best[1]:
                                best = (tname, mae, t.get("tolerance", DEFAULT_TOLERANCE))
                    if best is not None:
                        bname, bmae, btol = best
                        # compute score (1.0 means perfect, 0 at tol)
                        score = max(0.0, 1.0 - (bmae / btol)) if btol > 0 else 0.0
                        match_name, match_mae, match_score = bname, bmae, score

                # draw landmarks (visual debug)
                mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0,128,255), thickness=1))

            now = time.monotonic()

            # ---- emotion switching (improved) ----
            # when a new candidate appears, reset candidate_since
            if match_name != candidate_name:
                candidate_name = match_name
                candidate_since = now
            else:
                # same candidate continuing
                if candidate_name is not None:
                    # if high-confidence -> instant switch
                    if match_score is not None and match_score >= INSTANT_CONFIDENCE:
                        if displayed_emotion != candidate_name:
                            displayed_emotion = candidate_name
                            # make candidate_since now so we don't immediately revert
                            candidate_since = now
                    # else require stability duration
                    elif displayed_emotion != candidate_name and (now - candidate_since) >= SWITCH_THRESHOLD:
                        displayed_emotion = candidate_name
                else:
                    # no candidate detected -> clear after stability
                    if displayed_emotion is not None and (now - candidate_since) >= SWITCH_THRESHOLD:
                        displayed_emotion = None

            # ---- speaking switching (debounced) ----
            if speaking != speaking_state:
                if speaking_since is None:
                    speaking_since = now
                elif (now - speaking_since) >= SPEAK_SWITCH_THRESHOLD:
                    speaking_state = speaking
                    speaking_since = None
            else:
                speaking_since = None

            # ---- overlay webcam debug info ----
            y = 20
            cv2.putText(frame, f"Templates: {', '.join(templates.keys()) or 'none'}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1); y+=22
            cv2.putText(frame, f"REC: {'ON' if recording else 'OFF'} (press 'r')", (10,y), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0) if recording else (0,180,255),2); y+=26
            if match_name:
                cv2.putText(frame, f"Instant match -> {match_name} mae={match_mae:.4f} score={(match_score*100) if match_score is not None else 0:.0f}%", (10,y), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2); y+=26
            else:
                cv2.putText(frame, "Instant match -> none", (10,y), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,180,255),2); y+=26
            cv2.putText(frame, f"Displayed emotion: {displayed_emotion or 'none'}", (10,y), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2); y+=26
            cv2.putText(frame, f"Speaking: {'YES' if speaking_state else 'no'}", (10,y), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255) if speaking_state else (180,180,180),2); y+=26
            if recording:
                cv2.putText(frame, f"Recording samples: {len(record_samples)} (press 'r')", (10,y), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            cv2.imshow("Template Recorder Matcher (Smoothed)", frame)

            # ---- PNG Tuber window (resize to fit window) ----
            tuber_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH,4), dtype=np.uint8)
            tuber_frame[:,:] = GREEN_KEY

            emotion = displayed_emotion or "neutral"
            state = "speaking" if speaking_state and emotion in loaded_textures and "speaking" in loaded_textures[emotion] else "main"
            texture = loaded_textures.get(emotion, {}).get(state)

            if texture is not None:
                th, tw = texture.shape[:2]
                # scale to fill window while preserving aspect ratio:
                scale = min(WINDOW_WIDTH / tw, WINDOW_HEIGHT / th)
                new_w, new_h = max(1, int(tw * scale)), max(1, int(th * scale))
                resized = cv2.resize(texture, (new_w, new_h), interpolation=cv2.INTER_AREA)
                x = (WINDOW_WIDTH - new_w) // 2
                y = (WINDOW_HEIGHT - new_h) // 2
                y1, y2 = y, y + new_h
                x1, x2 = x, x + new_w
                alpha_s = resized[:, :, 3:4] / 255.0
                alpha_l = 1.0 - alpha_s
                tuber_frame[y1:y2, x1:x2, :3] = alpha_s * resized[:, :, :3] + alpha_l * tuber_frame[y1:y2, x1:x2, :3]

            cv2.imshow("PNG Tuber", tuber_frame)

            # ---- keyboard handling ----
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # toggle record: starts/stops and asks interval when starting
            if key == ord('r'):
                recording = not recording
                if recording:
                    record_samples = []
                    interval_input = input("Recording interval in seconds (e.g. 0.5): ").strip()
                    try:
                        record_interval = float(interval_input)
                    except:
                        record_interval = 0.0
                    last_record_time = 0.0
                    print(f"[record] started with interval {record_interval}s. Press 'r' to stop.")
                else:
                    if not record_samples:
                        print("[record] stopped but no samples recorded.")
                    else:
                        name = input("Template name (single word): ").strip()
                        if not name:
                            print("Aborted save.")
                        else:
                            tol_input = input(f"Tolerance MAE [default {DEFAULT_TOLERANCE}]:").strip()
                            try:
                                tol = float(tol_input) if tol_input else DEFAULT_TOLERANCE
                            except:
                                tol = DEFAULT_TOLERANCE
                            path = os.path.join(TEMPLATES_DIR, f"{name}.json")
                            if os.path.exists(path):
                                overwrite = input(f"Template '{name}' exists. Overwrite? (y/N):").strip().lower()
                                if overwrite != 'y':
                                    print("Aborted save.")
                                else:
                                    save_template(name, record_samples, tol)
                                    templates = load_templates()
                            else:
                                save_template(name, record_samples, tol)
                                templates = load_templates()
                    record_samples = []

            # reload textures + settings live
            if key == ord('l'):
                reload_settings_and_textures()

            # record samples on interval while recording
            if recording and live_vector is not None:
                if (now - last_record_time) >= record_interval:
                    record_samples.append(live_vector.tolist())
                    last_record_time = now

    finally:
        cap.release()
        cv2.destroyAllWindows()
        stream.stop()
