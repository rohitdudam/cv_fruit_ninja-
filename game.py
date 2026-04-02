import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random
import math
import numpy as np
from collections import deque
#hello this is rohit tesiting

# --- 1. AUTO-DOWNLOAD THE AI MODEL ---
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading the new MediaPipe AI Model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# --- 2. INITIALIZE NEW MEDIAPIPE TASKS API ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. ADVANCED MATH & RENDER HELPERS ---
def draw_transparent(background, overlay, x, y):
    """Safely draws PNGs even if they go off-screen."""
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay.shape

    y1, y2 = max(0, y), min(bg_h, y + h)
    x1, x2 = max(0, x), min(bg_w, x + w)
    
    y1o, y2o = max(0, -y), min(h, bg_h - y)
    x1o, x2o = max(0, -x), min(w, bg_w - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o: return background

    if overlay.shape[2] == 4:
        alpha = overlay[y1o:y2o, x1o:x2o, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            background[y1:y2, x1:x2, c] = (alpha * overlay[y1o:y2o, x1o:x2o, c] +
                                           alpha_inv * background[y1:y2, x1:x2, c])
    else:
        background[y1:y2, x1:x2] = overlay[y1o:y2o, x1o:x2o, :3]
    return background

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Calculates the distance from a fruit to the solid line of your blade swipe."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t)) 
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(px - closest_x, py - closest_y)

# --- 4. SETUP FRUIT IMAGES ---
IMAGE_FILES = ['apple.png', 'orange.png', 'strawberry.png']
FRUIT_RESOURCES = []

for path in IMAGE_FILES:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cv2.resize(img, (80, 80)) 
        FRUIT_RESOURCES.append(img)
    else:
        print(f"Warning: Could not find '{path}'. Using fallback circles.")

# --- 5. GAME VARIABLES ---
score = 0
missed = 0
game_over = False # NEW: Tracks if the game is currently over
fruits = []
cut_halves = []
floating_texts = [] 
trail = deque(maxlen=15) 

def spawn_fruit(w, h):
    img_idx = random.randint(0, len(FRUIT_RESOURCES)-1) if FRUIT_RESOURCES else None
    return {
        'x': random.randint(100, w - 100),
        'y': h + 50, 
        'vx': random.uniform(-4, 4),   
        'vy': random.uniform(-28, -36), 
        'radius': 40,
        'img_idx': img_idx
    }

def reset_game():
    """Resets all variables to start a fresh game."""
    global score, missed, game_over, fruits, cut_halves, floating_texts, trail
    score = 0
    missed = 0
    game_over = False
    fruits.clear()
    cut_halves.clear()
    floating_texts.clear()
    trail.clear()

# --- 6. START WEBCAM ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Game Loading... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Process Hand Tracking (Always tracking, even on game over screen)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        index_finger = hand[8]
        cx, cy = int(index_finger.x * w), int(index_finger.y * h)
        trail.appendleft((cx, cy))
    else:
        if len(trail) > 0: trail.pop()

    # ==========================================
    # --- ACTIVE GAME LOGIC ---
    # ==========================================
    if not game_over:
        # Spawn Fruits
        if random.randint(1, 100) <= 5:
            fruits.append(spawn_fruit(w, h))

        # --- PROCESS WHOLE FRUITS ---
        for fruit in fruits[:]: 
            fruit['x'] += fruit['vx']
            fruit['y'] += fruit['vy']
            fruit['vy'] += 0.8 

            # Draw Whole Fruit
            if fruit['img_idx'] is not None:
                fruit_img = FRUIT_RESOURCES[fruit['img_idx']]
                draw_x = int(fruit['x'] - 40)
                draw_y = int(fruit['y'] - 40)
                frame = draw_transparent(frame, fruit_img, draw_x, draw_y)
            else:
                cv2.circle(frame, (int(fruit['x']), int(fruit['y'])), 40, (0,0,255), -1)

            # Collision Detection
            is_cut = False
            if len(trail) > 0:
                if math.hypot(trail[0][0] - fruit['x'], trail[0][1] - fruit['y']) < 50:
                    is_cut = True

            if not is_cut:
                for i in range(1, len(trail)):
                    p1, p2 = trail[i-1], trail[i]
                    dist = point_to_segment_dist(fruit['x'], fruit['y'], p1[0], p1[1], p2[0], p2[1])
                    if dist < 50:  
                        is_cut = True
                        break

            if is_cut:
                score += 100
                floating_texts.append({'x': fruit['x'] - 20, 'y': fruit['y'], 'text': '+100', 'timer': 20})

                if fruit['img_idx'] is not None:
                    f_img = FRUIT_RESOURCES[fruit['img_idx']]
                    img_w = f_img.shape[1]
                    left_img = f_img[:, :img_w//2]
                    right_img = f_img[:, img_w//2:]
                    
                    cut_halves.append({
                        'x': fruit['x'] - 20, 'y': fruit['y'], 'vx': fruit['vx'] - 6, 'vy': fruit['vy'] - 2, 'img': left_img
                    })
                    cut_halves.append({
                        'x': fruit['x'] + 20, 'y': fruit['y'], 'vx': fruit['vx'] + 6, 'vy': fruit['vy'] - 2, 'img': right_img
                    })
                else:
                    cut_halves.append({'x': fruit['x']-20, 'y': fruit['y'], 'vx': -6, 'vy': -2, 'img': None})
                    cut_halves.append({'x': fruit['x']+20, 'y': fruit['y'], 'vx': 6, 'vy': -2, 'img': None})
                
                fruits.remove(fruit)
                continue

            # Check if fruit fell off screen
            if fruit['y'] > h + 100:
                missed += 1
                fruits.remove(fruit)
                # TRIGGER GAME OVER
                if missed >= 3:
                    game_over = True

        # --- PROCESS CUT HALVES ---
        for half in cut_halves[:]:
            half['x'] += half['vx']
            half['y'] += half['vy']
            half['vy'] += 0.8 
            
            if half['img'] is not None:
                draw_x = int(half['x'] - half['img'].shape[1]//2)
                draw_y = int(half['y'] - half['img'].shape[0]//2)
                frame = draw_transparent(frame, half['img'], draw_x, draw_y)
            else:
                cv2.circle(frame, (int(half['x']), int(half['y'])), 20, (0,0,255), -1)

            if half['y'] > h + 100:
                cut_halves.remove(half)

        # --- PROCESS FLOATING TEXT ANIMATIONS ---
        for ft in floating_texts[:]:
            cv2.putText(frame, ft['text'], (int(ft['x']), int(ft['y'])), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 4)
            cv2.putText(frame, ft['text'], (int(ft['x']), int(ft['y'])), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
            ft['y'] -= 3 
            ft['timer'] -= 1
            if ft['timer'] <= 0:
                floating_texts.remove(ft)

        # --- DRAW THE BLADE TRAIL ---
        for i in range(1, len(trail)):
            thickness = int(max(1, 15 - i * 1.5))
            inner_thickness = max(1, thickness // 2) 
            cv2.line(frame, trail[i - 1], trail[i], (255, 255, 255), thickness)
            cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), inner_thickness)

        # --- DRAW ACTIVE HUD ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.line(frame, (0, 80), (w, 80), (200, 200, 200), 2)

        score_text = f"SCORE: {score}"
        cv2.putText(frame, score_text, (30, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5) 
        cv2.putText(frame, score_text, (30, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2) 

        title_text = "NINJA AIR BLADE"
        text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_TRIPLEX, 1.2, 2)[0]
        title_x = (w - text_size[0]) // 2
        cv2.putText(frame, title_text, (title_x, 55), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5) 
        cv2.putText(frame, title_text, (title_x, 55), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 200, 255), 2) 

        miss_text = f"MISSED: {missed} / 3"
        text_size_miss = cv2.getTextSize(miss_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        miss_x = w - text_size_miss[0] - 30
        cv2.putText(frame, miss_text, (miss_x, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 5) 
        cv2.putText(frame, miss_text, (miss_x, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2) 

    # ==========================================
    # --- GAME OVER SCREEN LOGIC ---
    # ==========================================
    else:
        # 1. Darken the entire screen
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 2. Draw "GAME OVER"
        go_text = "GAME OVER"
        go_size = cv2.getTextSize(go_text, cv2.FONT_HERSHEY_TRIPLEX, 3.0, 5)[0]
        cv2.putText(frame, go_text, ((w - go_size[0]) // 2, h // 2 - 50), cv2.FONT_HERSHEY_TRIPLEX, 3.0, (0, 0, 0), 10)
        cv2.putText(frame, go_text, ((w - go_size[0]) // 2, h // 2 - 50), cv2.FONT_HERSHEY_TRIPLEX, 3.0, (0, 0, 255), 5)

        # 3. Draw Final Score
        final_score_text = f"FINAL SCORE: {score}"
        fs_size = cv2.getTextSize(final_score_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
        cv2.putText(frame, final_score_text, ((w - fs_size[0]) // 2, h // 2 + 30), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 6)
        cv2.putText(frame, final_score_text, ((w - fs_size[0]) // 2, h // 2 + 30), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)

        # 4. Draw Restart Instructions
        restart_text = "Press 'R' to Restart or 'Q' to Quit"
        r_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(frame, restart_text, ((w - r_size[0]) // 2, h // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(frame, restart_text, ((w - r_size[0]) // 2, h // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Still draw the blade trail so you can swish your hand around while dead!
        for i in range(1, len(trail)):
            thickness = int(max(1, 15 - i * 1.5))
            inner_thickness = max(1, thickness // 2) 
            cv2.line(frame, trail[i - 1], trail[i], (255, 255, 255), thickness)
            cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), inner_thickness)

    # Show the final compiled frame
    cv2.imshow("CV Fruit Ninja", frame)

    # Check for inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and game_over:
        reset_game()

cap.release()
cv2.destroyAllWindows()