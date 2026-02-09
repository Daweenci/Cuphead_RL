import mss
import numpy as np
import cv2
import pydirectinput
import time
import random
import os
import csv

# -------- SETTINGS --------
RUN_NAME = "test_run"  # change later for experiments
STEP_DELAY = 0.05          # 20 FPS
RUN_DURATION = 10          # seconds (change this!)
FRAME_SAVE_INTERVAL = 4    # save every x frame

hp_region = {"top": 1000, "left": 30, "width": 120, "height": 60}

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

    # Release keys not needed anymore
    for key in list(held_keys):
        if key not in target_keys:
            pydirectinput.keyUp(key)
            held_keys.remove(key)

    # Press new keys
    for key in target_keys:
        if key not in held_keys:
            pydirectinput.keyDown(key)
            held_keys.add(key)

print("Switch to Cuphead in 5 seconds...")
time.sleep(5)
print("Go!")

start_time = time.time()
step = 0

with mss.mss() as sct:
    monitor = sct.monitors[1]

    try:
        while True:

            # Stop after RUN_DURATION seconds
            if time.time() - start_time > RUN_DURATION:
                print("Time limit reached.")
                break

            # -------- OBSERVATION --------
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            img_small = cv2.resize(img_gray, (320, 180))

            # Debug view
            display = cv2.resize(img_small, (640, 360), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Agent View", display)

            # -------- HP REGION --------
            hp_screenshot = sct.grab(hp_region)
            hp_img = np.array(hp_screenshot)
            hp_gray = cv2.cvtColor(hp_img, cv2.COLOR_BGRA2GRAY)

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

            # Reassert shooting every loop
            if shoot:
                pydirectinput.keyDown("s")

            if dash:
                pydirectinput.press("shift")

            # -------- LOGGING --------
            if step % FRAME_SAVE_INTERVAL == 0:
                frame_path = os.path.join(frames_dir, f"frame_{step:06d}.png")
                cv2.imwrite(frame_path, img_small)

                csv_writer.writerow([step, move, crouch, shoot, dash, time.time()])


            # -------- TIMING --------
            time.sleep(STEP_DELAY)
            step += 1

    finally:
        print("Total steps:", step)
        release_all_keys()
        csv_file.close()
        cv2.destroyAllWindows()
