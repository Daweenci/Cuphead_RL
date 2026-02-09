import mss
import numpy as np
import cv2
import pydirectinput
import time
import random
import os
import csv
from dotenv import load_dotenv

load_dotenv()

# -------- SETTINGS --------
RUN_NAME = os.getenv("RUN_NAME")
STEP_DELAY = float(os.getenv("STEP_DELAY"))
RUN_DURATION = float(os.getenv("RUN_DURATION"))
FRAME_SAVE_INTERVAL = int(os.getenv("FRAME_SAVE_INTERVAL"))

print("Config:", RUN_NAME, STEP_DELAY, RUN_DURATION, FRAME_SAVE_INTERVAL)

hp_region = {"top": 1000, "left": 30, "width": 120, "height": 60}

# -------- LOAD HP TEMPLATES --------
template_paths = {
    4: ["templates/hp_4.png"],
    3: ["templates/hp_3.png"],
    2: ["templates/hp_2.png"],
    1: [
        "templates/hp_white_1.png",
        "templates/hp_red_1.png",
        "templates/hp_flicker_1.png"
    ],
    0: ["templates/hp_dead.png"]
}

templates = {}

for hp_value, paths in template_paths.items():
    templates[hp_value] = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        templates[hp_value].append(img)

last_damage_time = 0
DAMAGE_COOLDOWN = 1.0

def detect_hp_from_template(hp_gray):
    best_score = -1
    best_hp = None

    for hp_value, template_list in templates.items():
        for tmpl in template_list:
            res = cv2.matchTemplate(hp_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            score = res.max()

            if score > best_score:
                best_score = score
                best_hp = hp_value

    return best_hp, best_score

# -------- LOGGING SETUP --------
run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join("logs", f"{RUN_NAME}_{run_timestamp}")
frames_dir = os.path.join(run_dir, "frames")

os.makedirs(frames_dir, exist_ok=True)

csv_file = open(os.path.join(run_dir, "actions.csv"), "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["step", "move", "crouch", "shoot", "dash", "time"])

# -------- INPUT CONTROL --------
held_keys = set()
pydirectinput.PAUSE = 0

def release_all_keys():
    for key in list(held_keys):
        pydirectinput.keyUp(key)
    held_keys.clear()

def update_held_keys(target_keys):
    global held_keys

    for key in list(held_keys):
        if key not in target_keys:
            pydirectinput.keyUp(key)
            held_keys.remove(key)

    for key in target_keys:
        if key not in held_keys:
            pydirectinput.keyDown(key)
            held_keys.add(key)

print("Switch to Cuphead in 5 seconds...")
time.sleep(5)
print("Go!")

start_time = time.time()
step = 0

last_hp = 4
current_hp = 4

with mss.mss() as sct:
    monitor = sct.monitors[1]

    try:
        while True:

            if time.time() - start_time > RUN_DURATION:
                print("Time limit reached.")
                break

            # -------- OBSERVATION --------
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            img_small = cv2.resize(img_gray, (320, 180))

            display = cv2.resize(img_small, (640, 360), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Agent View", display)

            # -------- HP REGION --------
            hp_screenshot = sct.grab(hp_region)
            hp_img = np.array(hp_screenshot)
            hp_gray = cv2.cvtColor(hp_img, cv2.COLOR_BGRA2GRAY)

            detected_hp, score = detect_hp_from_template(hp_gray)

            now = time.time()

            if detected_hp is not None:

                # Only allow HP to decrease if cooldown passed
                if detected_hp < last_hp:

                    if now - last_damage_time > DAMAGE_COOLDOWN:
                        # Force exactly -1 damage
                        current_hp = last_hp - 1
                        last_damage_time = now
                        print("Damage confirmed (cooldown passed)")
                    else:
                        # Ignore flicker damage
                        current_hp = last_hp

                else:
                    # Same HP or higher (ignore increases)
                    current_hp = last_hp
            else:
                current_hp = last_hp


            if current_hp < last_hp:
                print("Damage taken!")

            if current_hp == 0 and last_hp != 0:
                print("Episode finished (HP reached 0)")

            last_hp = current_hp

            print("HP detected:", current_hp, "score:", round(score, 3))

            # -------- DEATH SAFETY --------
            if current_hp == 0:
                print("DEAD detected - stopping inputs")
                release_all_keys()
                time.sleep(STEP_DELAY)
                step += 1
                continue

            hp_display = cv2.resize(hp_gray, (240, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("HP View", hp_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # -------- RANDOM POLICY --------
            move = random.choice(["left", "right", None])
            crouch = random.random() < 0.3
            shoot = True
            dash = random.random() < 0.05

            target_keys = set()

            if move:
                target_keys.add(move)

            if crouch:
                target_keys.add("down")

            if shoot:
                target_keys.add("s")

            update_held_keys(target_keys)

            if shoot:
                pydirectinput.keyDown("s")

            if dash:
                pydirectinput.press("shift")

            # -------- LOGGING --------
            if step % FRAME_SAVE_INTERVAL == 0:
                frame_path = os.path.join(frames_dir, f"frame_{step:06d}.png")
                cv2.imwrite(frame_path, img_small)
                csv_writer.writerow([step, move, crouch, shoot, dash, time.time()])

            time.sleep(STEP_DELAY)
            step += 1

    finally:
        print("Total steps:", step)
        release_all_keys()
        csv_file.close()
        cv2.destroyAllWindows()
