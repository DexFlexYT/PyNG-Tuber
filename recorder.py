import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import math

# ----------------- config / defaults -----------------
# Directory to save the landmark templates
TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Default tolerance for matching (used when saving)
DEFAULT_TOLERANCE = 0.06

mp_face_mesh = mp.solutions.face_mesh

# Eye indices for normalization reference
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

# ----------------- template / matching helpers (kept for saving logic) -----------------

def save_template(name, samples, tolerance=DEFAULT_TOLERANCE):
    """Saves the recorded samples as a JSON template."""
    # Convert numpy arrays in samples back to lists for JSON serialization
    serializable_samples = [s.tolist() for s in samples]
    
    data = {
        "name": name, 
        "samples": serializable_samples, 
        "landmark_count": int(len(serializable_samples[0])//2) if serializable_samples else 0, 
        "tolerance": float(tolerance), 
        "timestamp": time.time()
    }
    path = os.path.join(TEMPLATES_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n[saved] template '{name}' ({len(samples)} samples) -> {path}")

def get_landmark_array(landmarks):
    """Converts MediaPipe landmarks to an Nx3 numpy array (x, y, z)."""
    return np.array([[lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0.0] for lm in landmarks], dtype=np.float32)

def normalize_landmarks_no_rotate(pts):
    """
    Normalizes landmarks by centering them and scaling by inter-eye distance (IOD).
    Returns a flattened list of 2D coordinates [x0, y0, x1, y1, ...]
    """
    if pts.shape[0] == 0:
        return None
    
    center = pts.mean(axis=0)
    pts_c = pts - center
    
    # Use IOD for scaling if the required landmarks are present
    if pts.shape[0] > max(LEFT_EYE_IDX, RIGHT_EYE_IDX):
        left = pts[LEFT_EYE_IDX][:2]
        right = pts[RIGHT_EYE_IDX][:2]
        iod = np.linalg.norm(right - left)
    else:
        # Fallback scaling if eye indices are missing
        iod = np.max(np.linalg.norm(pts_c[:, :2], axis=1))
    
    if iod <= 1e-6:
        iod = 1.0
    
    # Normalize by IOD (or fallback scale)
    pts_n = pts_c[:, :2] / iod
    
    return pts_n.flatten().tolist()

# ----------------- head rotation / correction (for stable templates) -----------------
# We keep the head rotation logic to ensure the recorded templates are normalized
# for rotation, making the resulting emotion template stable regardless of 
# the slight head tilt during recording.

NOSE_IDX = 1
CHIN_IDX = 152

def compute_head_rotation(pts3):
    """Estimates Euler angles (pitch, yaw, roll) from key face points."""
    try:
        left = pts3[LEFT_EYE_IDX]
        right = pts3[RIGHT_EYE_IDX]
        nose = pts3[NOSE_IDX]
        # Use nose and eye line to compute roll/yaw/pitch estimate
        
        eye_line = right - left
        nose_line = np.array([0, 0, 1]) # Simplified reference normal (Z-axis)
        
        # This function is simplified from the original but keeps the core idea:
        # roll: angle of eye_line in the XY plane
        roll = math.degrees(math.atan2(eye_line[1], eye_line[0]))
        
        # A more robust 3D rotation estimate requires a full PnP solver or more points,
        # but for the template normalization goal, a simplified normal calculation works.
        # For simplicity and mirroring the original script's *intent* to correct rotation:
        try:
            chin = pts3[CHIN_IDX]
            nose_chin = chin - nose
            normal = np.cross(eye_line, nose_chin)
            normal = normal / (np.linalg.norm(normal)+1e-8)
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, -normal[1]))))
            yaw = math.degrees(math.asin(max(-1.0, min(1.0, normal[0]))))
        except IndexError:
            pitch, yaw = 0.0, 0.0 # Fallback
        
        return {'pitch': float(pitch), 'yaw': float(yaw), 'roll': float(roll)}
    except IndexError:
        return {'pitch':0.0, 'yaw':0.0, 'roll':0.0}

def euler_to_matrix_3d(p_rad, y_rad, r_rad):
    """Converts Euler angles to a 3D rotation matrix (Z-Y-X order)."""
    cp, sp = math.cos(p_rad), math.sin(p_rad)
    cy, sy = math.cos(y_rad), math.sin(y_rad)
    cr, sr = math.cos(r_rad), math.sin(r_rad)

    Rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])

    R = Rz @ Ry @ Rx
    return R

def invert_rotation_and_apply(points, pitch, yaw, roll):
    """Applies the inverse of the head rotation to the landmarks."""
    p_rad = math.radians(-pitch)
    y_rad = math.radians(-yaw)
    r_rad = math.radians(-roll)
    R_inv = euler_to_matrix_3d(p_rad, y_rad, r_rad) 

    center = points.mean(axis=0)

    pts_centered = points - center
    pts_rotated = (R_inv @ pts_centered.T).T
    pts_corrected = pts_rotated + center
    return pts_corrected

# ----------------- main loop -----------------
# runtime state
cap = cv2.VideoCapture(0)
recording = False
record_samples = []
record_interval = 0.0
last_record_time = 0.0

WINDOW_NAME = 'Face Landmark Recorder'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 480)

print("--- Face Landmark Recorder ---")
print("Press 'r' to start/stop recording a new template.")
print("Press 'q' or ESC to quit.")
print(f"Templates will be saved to: {os.path.abspath(TEMPLATES_DIR)}")
print("------------------------------")

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
            
            # Flip image for a selfie-view display
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            
            current_time = time.monotonic()
            
            # Draw on the frame and prepare data
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pts3 = get_landmark_array(lm)
                
                # Draw landmarks on the camera feed for visualization
                for landmark in lm:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # 1. Compute head rotation
                head_rot = compute_head_rotation(pts3)

                # 2. Apply inverse rotation for stability
                pts3_corrected = invert_rotation_and_apply(
                    pts3,
                    head_rot['pitch'],
                    head_rot['yaw'],
                    head_rot['roll']
                )

                # 3. Normalize landmarks to get a stable feature vector
                live_vector = normalize_landmarks_no_rotate(pts3_corrected)

                # 4. Recording logic
                if recording and live_vector is not None:
                    if current_time - last_record_time >= record_interval:
                        # Convert list back to numpy array for consistency before saving
                        live_vector_np = np.array(live_vector, dtype=np.float32)
                        record_samples.append(live_vector_np)
                        last_record_time = current_time
                        print(f"[record] Sampled {len(record_samples)} / {current_time:.2f}s", end='\r')


            # Display status on screen
            status_text = "Ready. Press 'r' to start."
            if recording:
                status_text = f"RECORDING... Samples: {len(record_samples)} (Interval: {record_interval}s)"
                
            cv2.putText(frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if recording else (255, 255, 255), 2)
            
            cv2.imshow(WINDOW_NAME, frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break
            
            if key == ord('r'):
                # Start recording
                if not recording:
                    record_samples = []
                    interval_input = input('Recording interval in seconds (e.g. 0.5): ').strip()
                    try:
                        record_interval = float(interval_input)
                        if record_interval <= 0:
                            print("[record] Interval must be positive, defaulting to 0.1s.")
                            record_interval = 0.1
                    except:
                        record_interval = 0.1
                        print("[record] Invalid input, defaulting to 0.1s.")

                    recording = True
                    last_record_time = 0.0
                    print(f"[record] STARTED with interval {record_interval}s. Press 'r' to STOP.")
                    
                # Stop recording and save
                else: 
                    recording = False
                    print('\n[record] STOPPED.')
                    
                    if not record_samples:
                        print('[record] Stopped but no samples recorded.')
                    else:
                        print(f"Collected {len(record_samples)} samples.")
                        
                        name = input('Enter **Template name** (single word, e.g., happy): ').strip()
                        if not name:
                            print('Aborted save. No name provided.')
                        else:
                            tol_input = input(f'Enter Tolerance MAE [default {DEFAULT_TOLERANCE}]:').strip()
                            try:
                                tol = float(tol_input) if tol_input else DEFAULT_TOLERANCE
                            except:
                                tol = DEFAULT_TOLERANCE
                            
                            save_template(name, record_samples, tol)
                        
                        record_samples = [] # Clear samples after saving/aborting save

    except Exception as e:
        print("[error]", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()

# The script does not require the audio stream or networking, so those are removed.