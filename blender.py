import bpy
import socket
import json
import threading
import time
from math import radians

# ---------------- Networking ----------------
STREAM_HOST = '127.0.0.1'
STREAM_PORT = 5005

stream_state = {
    'head_rotation': {'yaw':0.0, 'pitch':0.0, 'roll':0.0},
    'eye_open': 1.0,
    'brows': 0.0,
    'volume': 0.0,
    'emotion': 'neutral',
    'speaking': False,
    'confidence': 0.0
}

# ---------------- Settings ----------------
LERP_FACTOR = 0.3
NEUTRAL_LERP_FACTOR = 0.01
BROW_NEUTRAL_BLEND = 0.005  # stabilizer for brow neutral
BROWS_LERP = 0.2  # how smooth brows move
MOUTH_LERP = 1.0
PUPIL_LERP = 0.35
VOLUME_LERP = 0.85
EYE_OPEN_LERP = 0.7  # only when increasing

DEADZONE_DEG = 0.25
BODY_ROT_SCALE = 0.3
HEAD_ROT_SCALE = .6
EYE_ROT_MULTIPLIER = 5.0
EYE_X_MIN, EYE_X_MAX = -50.0, 50.0
EYE_Y_MIN, EYE_Y_MAX = -50.5, 50.5
ARMATURE_NAME = 'Player Armature'
HEAD_BONE_NAME = 'Bone.004'
BODY_BONE_NAME = 'Body'
FACEPLATE_OBJECT_NAME = 'faceplate'
MOUTH_IMAGE_NODE_NAME = 'Image Texture.003'
MOUTH_VALUE_NODE_NAME = 'Mouth'
EYE_X_NODE_NAME = 'EyeXPos'
EYE_Y_NODE_NAME = 'EyeYPos'
EYE_OPEN_NODE_NAME = 'Eyes'
BROWS_NODE_NAME = 'Brows'

EMOTION_TO_MOUTH_FRAME = {
    'neutral': 1,
    'happy': 12,
    'angry': 1,
    'surprised': 6,
    'mewing': 9,
    'peculiar': 5
}

# ---------------- Helpers ----------------
def lerp(a, b, f):
    return a + (b - a) * f

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def within_deadzone(a, b, threshold):
    return abs(a - b) < threshold

# ---------------- Networking Listener ----------------
def listen_stream_loop():
    global stream_state
    while True:
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            print(f"[listener] connecting to {STREAM_HOST}:{STREAM_PORT}...")
            s.connect((STREAM_HOST, STREAM_PORT))
            s.settimeout(1.0)
            buffer = ''
            while True:
                try:
                    data = s.recv(4096).decode('utf-8')
                except socket.timeout:
                    data = ''
                if not data:
                    continue
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        for key in ['head_rotation', 'eye_open', 'brows', 'volume', 'emotion', 'speaking', 'confidence']:
                            if key in d:
                                stream_state[key] = d[key]
                    except Exception as e:
                        print('[listener] parse error:', e)
        except Exception as e:
            print('[listener] connection failed:', e)
        finally:
            if s:
                try: s.close()
                except: pass
        time.sleep(1.0)

threading.Thread(target=listen_stream_loop, daemon=True).start()

# ---------------- Faceplate Helpers ----------------
def find_faceplate_object():
    if FACEPLATE_OBJECT_NAME:
        obj = bpy.data.objects.get(FACEPLATE_OBJECT_NAME)
        if obj: return obj
    arm = bpy.data.objects.get(ARMATURE_NAME)
    if arm:
        for child in arm.children:
            if child.type == 'MESH' and child.parent_type == 'BONE' and child.parent_bone == HEAD_BONE_NAME:
                return child
    return None

def get_node_by_name(mat, name):
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None
    return mat.node_tree.nodes.get(name)

faceplate_obj = find_faceplate_object()
faceplate_material = None
if faceplate_obj and faceplate_obj.material_slots:
    try:
        faceplate_material = faceplate_obj.material_slots[0].material
    except: pass

print("Faceplate:", faceplate_obj)
print("Material:", faceplate_material)

# ---------------- Rig ----------------
armature = bpy.data.objects.get(ARMATURE_NAME)
if armature:
    head_bone = armature.pose.bones.get(HEAD_BONE_NAME)
    body_bone = armature.pose.bones.get(BODY_BONE_NAME)
    if head_bone: head_bone.rotation_mode = 'XYZ'
    if body_bone: body_bone.rotation_mode = 'XYZ'
else:
    head_bone = body_bone = None

NEUTRAL_OFFSET = {'yaw':0.0, 'pitch':25.0, 'roll':3.0}
BROW_NEUTRAL = -0.025933
target_rot = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}


_current_eye_open = 1.0

_current_mouth_frame = float(EMOTION_TO_MOUTH_FRAME.get('neutral',0))
_current_mouth_open = 0.0
_current_eye_x = 0.5
_current_eye_y = 0.5
_current_volume = 0.0
_current_brows = 0.0

# ---------------- Update Handler ----------------
def update_all(scene):
    global target_rot, NEUTRAL_OFFSET
    global _current_mouth_frame, _current_mouth_open, _current_eye_x, _current_eye_y, _current_volume
    global _current_eye_open, _current_brows, BROW_NEUTRAL
    

    # Brow stabilizer + interpolation 
    brow_val = float(stream_state.get('brows', 0.0))
    BROW_NEUTRAL = lerp(BROW_NEUTRAL, brow_val, BROW_NEUTRAL_BLEND)
    brow_delta = brow_val - BROW_NEUTRAL
    _current_brows = lerp(_current_brows, brow_delta, BROWS_LERP)

    node_brows = get_node_by_name(faceplate_material, BROWS_NODE_NAME)
    if node_brows and node_brows.type == 'VALUE':
        node_brows.outputs[0].default_value = _current_brows


    # ---- Update rig ----
    arm = bpy.data.objects.get(ARMATURE_NAME)
    if arm:
        head = arm.pose.bones.get(HEAD_BONE_NAME)
        body = arm.pose.bones.get(BODY_BONE_NAME)
        if head and body:
            hd = stream_state['head_rotation']
            for k in NEUTRAL_OFFSET:
                NEUTRAL_OFFSET[k] = lerp(NEUTRAL_OFFSET[k], hd[k], NEUTRAL_LERP_FACTOR)
            corrected = {
                'pitch': hd['pitch'] - NEUTRAL_OFFSET['pitch'],
                'yaw': hd['yaw'] - NEUTRAL_OFFSET['yaw'],
                'roll': hd['roll'] - NEUTRAL_OFFSET['roll']
            }
            for k in corrected:
                if not within_deadzone(corrected[k], target_rot[k], DEADZONE_DEG):
                    target_rot[k] = corrected[k]
            head.rotation_euler[0] = radians(lerp(head.rotation_euler[0]*180/3.14159, target_rot['pitch']*HEAD_ROT_SCALE, LERP_FACTOR))
            head.rotation_euler[1] = radians(lerp(head.rotation_euler[1]*180/3.14159, target_rot['yaw']*HEAD_ROT_SCALE, LERP_FACTOR))
            head.rotation_euler[2] = radians(lerp(head.rotation_euler[2]*180/3.14159, target_rot['roll']*HEAD_ROT_SCALE, LERP_FACTOR))
            body.rotation_euler[0] = radians(lerp(body.rotation_euler[0]*180/3.14159, target_rot['pitch']*BODY_ROT_SCALE, LERP_FACTOR))
            body.rotation_euler[1] = radians(lerp(body.rotation_euler[1]*180/3.14159, target_rot['yaw']*BODY_ROT_SCALE, LERP_FACTOR))
            body.rotation_euler[2] = radians(lerp(body.rotation_euler[2]*180/3.14159, target_rot['roll']*BODY_ROT_SCALE, LERP_FACTOR))
            arm.update_tag()

    # ---- Update faceplate ----
    if faceplate_material and faceplate_material.use_nodes:
        mouth_frame_target = EMOTION_TO_MOUTH_FRAME.get(stream_state.get('emotion','neutral'),'neutral')
        _current_mouth_frame = lerp(_current_mouth_frame, mouth_frame_target, MOUTH_LERP)
        _current_mouth_open = lerp(_current_mouth_open, stream_state.get('volume',0.0)*2000, VOLUME_LERP)

        hr = stream_state.get('head_rotation', {'yaw':0,'pitch':0,'roll':0})
        x_delta = clamp(hr.get('yaw',0.0) * EYE_ROT_MULTIPLIER / 90.0, EYE_X_MIN, EYE_X_MAX)
        y_delta = clamp(-hr.get('pitch',0.0) * EYE_ROT_MULTIPLIER / 90.0, EYE_Y_MIN, EYE_Y_MAX)
        head_eye_x = 0.5 + x_delta
        head_eye_y = 0.5 + y_delta
        final_eye_x = head_eye_x
        final_eye_y = head_eye_y
        _current_eye_x = lerp(_current_eye_x, final_eye_x, PUPIL_LERP)
        _current_eye_y = lerp(_current_eye_y, final_eye_y, PUPIL_LERP)

        # ---- Eye openness with one-way interpolation ----
        target_eye_open = float(stream_state.get('eye_open', 1.0))
        if target_eye_open > _current_eye_open:
            _current_eye_open = lerp(_current_eye_open, target_eye_open, 0.3)
        else:
            _current_eye_open = target_eye_open


        # ---- Apply nodes ----
        node_img = get_node_by_name(faceplate_material, MOUTH_IMAGE_NODE_NAME)
        if node_img and getattr(node_img, 'image', None):
            node_img.image_user.frame_offset = int(round(_current_mouth_frame))
        node_val = get_node_by_name(faceplate_material, MOUTH_VALUE_NODE_NAME)
        if node_val and node_val.type=='VALUE':
            node_val.outputs[0].default_value = _current_mouth_open

        node_x = get_node_by_name(faceplate_material, EYE_X_NODE_NAME)
        node_y = get_node_by_name(faceplate_material, EYE_Y_NODE_NAME)
        if node_x and node_x.type=='VALUE': node_x.outputs[0].default_value = _current_eye_x
        if node_y and node_y.type=='VALUE': node_y.outputs[0].default_value = _current_eye_y

        node_open = get_node_by_name(faceplate_material, EYE_OPEN_NODE_NAME)
        if node_open and node_open.type=='VALUE':
            node_open.outputs[0].default_value = _current_eye_open

        node_brows = get_node_by_name(faceplate_material, BROWS_NODE_NAME)
        if node_brows and node_brows.type=='VALUE':
            node_brows.outputs[0].default_value = _current_brows

# ---------------- Register Handler ----------------
if update_all not in bpy.app.handlers.frame_change_post:
    bpy.app.handlers.frame_change_post.append(update_all)
    print("[blender] rig + faceplate live update registered")
