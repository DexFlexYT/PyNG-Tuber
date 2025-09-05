import cv2
from deepface import DeepFace
from collections import deque, Counter, defaultdict
import os
import numpy as np
import time  # Add at the top

# Webcam
cap = cv2.VideoCapture(0)

# Rolling history for smoothing
history = deque(maxlen=10)
# Path to textures (PNG images for emotions)
TEXTURES_PATH = "textures/placeholders"

# Mapping of base emotions to filenames
EMOTION_IMAGES = {
    "angry": "angry.png",
    "disgust": "disgust.png",
    "fear": "fear.png",
    "happy": "happy.png",
    "sad": "sad.png",
    "surprise": "surprise.png",
    "neutral": "neutral.png"
}

# Custom emotions
CUSTOM_EMOTIONS = {
    "mewing": [
        ("angry", 2, 35),
        ("fear", 12, 69),
        ("happy", 0, 20),
        ("sad", 40, 70)
    ],
    "mewing_alt": [
        ("angry", 0, 13),
        ("fear", 0, 17),
        ("happy", 0, 10),
        ("sad", 43, 98),
        ("neutral", 0, 41),
    ]
}
EMOTION_IMAGES.update({"mewing": "mewing.png"})
EMOTION_IMAGES.update({"mewing_alt": "mewing_alt.png"})

# Preload textures
textures = {}
for emotion, filename in EMOTION_IMAGES.items():
    path = os.path.join(TEXTURES_PATH, filename)
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        textures[emotion] = img

# Recording state
recording = False
recorded_data = []

def overlay_transparent(background, overlay, x, y, scale=1.0):
    overlay = cv2.resize(overlay, (int(overlay.shape[1]*scale), int(overlay.shape[0]*scale)))
    h, w = overlay.shape[:2]
    x_end = min(x + w, background.shape[1])
    y_end = min(y + h, background.shape[0])
    w_clip = x_end - x
    h_clip = y_end - y
    if w_clip <= 0 or h_clip <= 0:
        return background
    overlay_rgb = overlay[:h_clip, :w_clip, :3] if overlay.shape[2]>=3 else overlay[:h_clip, :w_clip]
    if overlay.shape[2] == 4:
        alpha = overlay[:h_clip, :w_clip, 3:] / 255.0
    else:
        alpha = np.ones((h_clip, w_clip, 1), dtype=float)
    background[y:y+h_clip, x:x+w_clip] = (alpha * overlay_rgb + (1-alpha) * background[y:y+h_clip, x:x+w_clip]).astype(np.uint8)
    return background

def detect_custom_emotion(emotions: dict):
    for custom, conditions in CUSTOM_EMOTIONS.items():
        if all(minp <= emotions.get(base,0) <= maxp for base,minp,maxp in conditions):
            return custom
    return None

# Button area
button_top_left = (10, 10)
button_bottom_right = (180, 50)

def mouse_callback(event, x, y, flags, param):
    global recording
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_top_left[0] <= x <= button_bottom_right[0] and button_top_left[1] <= y <= button_bottom_right[1]:
            recording = not recording
            if not recording:
                if recorded_data:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename_txt = f"emotion_record_{timestamp}.txt"

                    all_emotions = recorded_data[0].keys()
                    averages = {k: np.mean([frame[k] for frame in recorded_data]) for k in all_emotions}
                    ranges = {k: (np.min([frame[k] for frame in recorded_data]), np.max([frame[k] for frame in recorded_data])) for k in all_emotions}

                    # Write the main data file
                    with open(filename_txt, "w") as f:
                        for i, frame_data in enumerate(recorded_data):
                            f.write(f"Frame {i+1}:\n")
                            for k, v in frame_data.items():
                                f.write(f"  {k}: {v:.2f}\n")
                        f.write("\nAverages:\n")
                        for k, v in averages.items():
                            f.write(f"  {k}: {v:.2f}\n")
                        f.write("\nRanges:\n")
                        for k, (minv, maxv) in ranges.items():
                            f.write(f"  {k}: {minv:.2f} - {maxv:.2f}\n")

                        # --- Custom-emotion-style section ---
                        # --- Custom-emotion-style block ---
                        f.write("\n# Generated recorded-emotion ranges:\n")
                        f.write("\"recorded_emotion\": [\n")
                        for k, (minv, maxv) in ranges.items():
                            f.write(f"    (\"{k}\", {minv:.0f}, {maxv:.0f}),\n")
                        f.write("]\n")


                    recorded_data.clear()
                    print(f"Saved emotion data to {filename_txt}")

cv2.namedWindow("Webcam Debug")
cv2.setMouseCallback("Webcam Debug", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    try:
        result = DeepFace.analyze(frame, actions=['emotion'],
                                  enforce_detection=False,
                                  detector_backend='opencv')[0]
        emotions = result["emotion"]
        dominant = result["dominant_emotion"]

        custom = detect_custom_emotion(emotions)
        if custom:
            dominant = custom

        history.append(dominant)
        smoothed = Counter(history).most_common(1)[0][0]

        # Greenscreen window
        green_bg = np.zeros((480, 640, 3), dtype=np.uint8)
        green_bg[:] = (0, 255, 0)
        if smoothed in textures:
            tex = textures[smoothed]
            x = max((green_bg.shape[1] - tex.shape[1]) // 2, 0)
            y = max((green_bg.shape[0] - tex.shape[0]) // 2, 0)
            green_bg = overlay_transparent(green_bg, tex, x, y)
        cv2.imshow("VTuber Window", green_bg)

        # Draw button
        cv2.rectangle(frame, button_top_left, button_bottom_right, (0,0,255) if not recording else (0,255,0), -1)
        cv2.putText(frame, "REC" if recording else "START REC", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

        # Emotion percentages (one per line)
        rect_height = 20 + 20*len(emotions)
        cv2.rectangle(frame, (0, frame.shape[0]-rect_height), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        for i,(k,v) in enumerate(emotions.items()):
            cv2.putText(frame, f"{k}: {int(v)}", (10, frame.shape[0]-rect_height+20+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        cv2.imshow("Webcam Debug", frame)

        # Record frame if active
        if recording:
            recorded_data.append({k:v for k,v in emotions.items()})

    except Exception as e:
        print("Error:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
