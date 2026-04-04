import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# ---------------- CONFIG ----------------
SMOOTHING = 0.8
SHOW_DEBUG = True
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS = 1
# ----------------------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
prev_rotations = {}

def smooth(name, rot):
    if name not in prev_rotations:
        prev_rotations[name] = rot
        return rot

    prev = prev_rotations[name]
    slerp = Slerp([0, 1], R.from_quat([prev.as_quat(), rot.as_quat()]))
    blended = slerp([SMOOTHING])[0]
    prev_rotations[name] = blended
    return blended

def safe_norm(v):
    n = np.linalg.norm(v)
    return None if n < 1e-6 else v / n

def rotation_from_to(v0, v1):
    v0 = safe_norm(v0)
    v1 = safe_norm(v1)
    if v0 is None or v1 is None:
        return R.identity()

    axis = np.cross(v0, v1)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    angle = np.arccos(dot)

    if np.linalg.norm(axis) < 1e-6:
        return R.identity()

    return R.from_rotvec(axis / np.linalg.norm(axis) * angle)

# Reference directions (T-pose assumption)
REF_UPPER = np.array([1.0, 0.0, 0.0])
REF_LOWER = np.array([1.0, 0.0, 0.0])

def extract_arm(side, lm):
    s = lm[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value]
    e = lm[mp_pose.PoseLandmark[f"{side}_ELBOW"].value]
    w = lm[mp_pose.PoseLandmark[f"{side}_WRIST"].value]

    shoulder = np.array([s.x, s.y, s.z])
    elbow    = np.array([e.x, e.y, e.z])
    wrist    = np.array([w.x, w.y, w.z])

    upper = elbow - shoulder
    lower = wrist - elbow

    shoulder_rot = smooth(f"{side}_shoulder", rotation_from_to(REF_UPPER, upper))
    elbow_rot    = smooth(f"{side}_elbow", rotation_from_to(REF_LOWER, lower))

    return shoulder_rot, elbow_rot, s, e

def draw_text_block(img, x, y, lines, color):
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * 14),
            FONT,
            FONT_SCALE,
            color,
            THICKNESS,
            cv2.LINE_AA
        )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        for side, color in (("LEFT", (0, 255, 0)), ("RIGHT", (255, 0, 0))):
            shoulder_rot, elbow_rot, s_lm, e_lm = extract_arm(side, lm)

            s_euler = shoulder_rot.as_euler("xyz", degrees=True)
            e_euler = elbow_rot.as_euler("xyz", degrees=True)

            sx, sy = int(s_lm.x * w), int(s_lm.y * h)
            ex, ey = int(e_lm.x * w), int(e_lm.y * h)

            draw_text_block(
                frame,
                sx + 5,
                sy - 10,
                [
                    f"{side} SHOULDER",
                    f"x {s_euler[0]:6.1f}",
                    f"y {s_euler[1]:6.1f}",
                    f"z {s_euler[2]:6.1f}",
                ],
                color
            )

            draw_text_block(
                frame,
                ex + 5,
                ey - 10,
                [
                    f"{side} ELBOW",
                    f"x {e_euler[0]:6.1f}",
                    f"y {e_euler[1]:6.1f}",
                    f"z {e_euler[2]:6.1f}",
                ],
                color
            )

        if SHOW_DEBUG:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

    cv2.imshow("Arm Rotation Test (On-Screen)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
