"""
Mediapipe multi-frame template recorder + rotation-invariant matcher
with time-based smoothing (800 ms).

Controls:
  r - toggle recording (start/stop). When stopping, you'll be prompted to name the template and set tolerance.
  q - quit

Templates saved to ./templates/<name>.json containing multiple samples.
Matching must be stable for 0.8s before the displayed emotion actually switches.
"""

import cv2
import mediapipe as mp
import numpy as np
import os, json, time
from collections import deque

TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

# Default tolerance (MAE) — lowered for stricter matching
DEFAULT_TOLERANCE = 0.06

# Smoothing threshold (seconds)
SWITCH_THRESHOLD = 0.4

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

def save_template(name, samples, tolerance=DEFAULT_TOLERANCE):
    data = {
        "name": name,
        "samples": samples,
        "landmark_count": int(len(samples[0])//2) if samples else 0,
        "tolerance": float(tolerance),
        "timestamp": time.time()
    }
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
                templates[data["name"]] = {
                    "samples": samples,
                    "landmark_count": int(data.get("landmark_count", 0)),
                    "tolerance": float(data.get("tolerance", DEFAULT_TOLERANCE)),
                    "path": path
                }
            except Exception as e:
                print("Failed to load template", path, e)
    return templates

def orthogonal_procrustes_mae(X, Y):
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
    except Exception:
        R = np.eye(2, dtype=np.float32)
    Xr = Xa @ R
    mae = float(np.mean(np.abs(Xr - Ya)))
    return mae

# prepare capture and templates
cap = cv2.VideoCapture(0)
templates = load_templates()
print("Loaded templates:", list(templates.keys()))

recording = False
record_samples = []

# smoothing state
candidate_name = None         # current instantaneous best match
candidate_since = None        # time.monotonic() when candidate started
displayed_emotion = None      # the smoothed/stable emotion shown to user

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

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

                # find best template/sample (minimal MAE)
                best = None
                for tname, t in templates.items():
                    for samp in t["samples"]:
                        try:
                            samp_pts = samp.reshape(-1, 2)
                            live_pts = live_vector.reshape(-1, 2)
                        except Exception:
                            samp_pts = samp.reshape(-1, 2)
                            live_pts = live_vector.reshape(-1, 2)
                        mae = orthogonal_procrustes_mae(live_pts, samp_pts)
                        if best is None or mae < best[1]:
                            best = (tname, mae, t.get("tolerance", DEFAULT_TOLERANCE))
                if best is not None:
                    bname, bmae, btol = best
                    if bmae <= btol:
                        match_name = bname
                        match_mae = bmae
                        match_score = max(0.0, 1.0 - (bmae / btol)) if btol > 0 else 0.0

            # debug draw
            mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0,128,255), thickness=1))

        # --- smoothing logic (800 ms threshold) ---
        now = time.monotonic()
        if match_name != candidate_name:
            # new instantaneous candidate detected (or None)
            candidate_name = match_name
            candidate_since = now if match_name is not None else now  # start counting for both on/off
        else:
            # candidate_name unchanged; check duration
            if candidate_name is not None:
                if displayed_emotion != candidate_name and (now - candidate_since) >= SWITCH_THRESHOLD:
                    displayed_emotion = candidate_name
            else:
                # candidate is None (no match). if displayed_emotion should clear after threshold:
                if displayed_emotion is not None and (now - candidate_since) >= SWITCH_THRESHOLD:
                    displayed_emotion = None

        # UI overlays
        y = 20
        status = f"Templates: {', '.join(templates.keys()) or 'none'}"
        cv2.putText(frame, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        y += 22

        rec_text = f"REC: {'ON' if recording else 'OFF'} (press 'r' to toggle)"
        cv2.putText(frame, rec_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if recording else (0,180,255), 2)
        y += 26

        # instantaneous match shown
        if match_name:
            cv2.putText(frame, f"Instant match -> {match_name}  mae={match_mae:.4f}  score={match_score*100:.0f}%",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            cv2.putText(frame, "Instant match -> none", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,180,255), 2)
        y += 26

        # displayed (smoothed) emotion
        cv2.putText(frame, f"Displayed (stable >= {int(SWITCH_THRESHOLD*1000)}ms): {displayed_emotion or 'none'}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 26

        # recording UI
        if recording:
            cv2.putText(frame, f"Recording samples: {len(record_samples)} (press 'r' to stop)",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Template Recorder Matcher (Smoothed)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r'):
            recording = not recording
            if recording:
                record_samples = []
                print("[record] started. Perform the expression, move your head around; press 'r' again to stop.")
            else:
                if not record_samples:
                    print("[record] stopped but no samples recorded.")
                else:
                    name = input("Template name (single word, e.g. happy): ").strip()
                    if name == "":
                        print("Aborted save (empty name).")
                    else:
                        path = os.path.join(TEMPLATES_DIR, f"{name}.json")
                        if os.path.exists(path):
                            overwrite = input(f"Template '{name}' exists. Overwrite? (y/N): ").strip().lower()
                            if overwrite != 'y':
                                print("Aborted save.")
                                continue
                        tol_input = input(f"Tolerance MAE (suggest 0.03..0.12) [default {DEFAULT_TOLERANCE}]: ").strip()
                        try:
                            tol = float(tol_input) if tol_input != "" else DEFAULT_TOLERANCE
                        except:
                            tol = DEFAULT_TOLERANCE
                        save_template(name, record_samples, tolerance=tol)
                        templates = load_templates()
                record_samples = []

        # append samples while recording
        if recording and (live_vector is not None):
            record_samples.append(live_vector.tolist())

    # end loop

cap.release()
cv2.destroyAllWindows()
