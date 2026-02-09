import pydirectinput
import time
import random

STEP_DELAY = 0.05  # 20 FPS

held_keys = set()

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

try:
    for step in range(200):

        # Random movement choice
        move = random.choice(["left", "right", None])

        # Random crouch
        crouch = random.random() < 0.3

        # Always shoot for now
        shoot = True

        # Random dash sometimes
        dash = random.random() < 0.1

        target_keys = set()

        if move:
            target_keys.add(move)

        if crouch:
            target_keys.add("down")

        if shoot:
            target_keys.add("s")

        update_held_keys(target_keys)

        if dash:
            pydirectinput.press("shift")

        time.sleep(STEP_DELAY)

finally:
    release_all_keys()
