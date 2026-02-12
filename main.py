import mss
import numpy as np
import cv2
import pydirectinput
import time
import random
import os
import csv
import yaml
import damage_tracker


# ============================================================
# CONFIG
# ============================================================

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    print(
        "Config:",
        cfg["run_name"],
        cfg["step_delay"],
        cfg["frame_save_interval"],
        cfg["num_episodes"]
    )
    return cfg


# ============================================================
# HP TEMPLATE HANDLING
# ============================================================

def load_hp_templates():
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

    return templates


def detect_hp_from_template(hp_gray, templates):
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


# ============================================================
# INPUT CONTROL
# ============================================================

held_keys = set()
pydirectinput.PAUSE = 0


def release_all_keys():
    for key in list(held_keys):
        pydirectinput.keyUp(key)
    held_keys.clear()


def update_held_keys(target_keys):
    for key in list(held_keys):
        if key not in target_keys:
            pydirectinput.keyUp(key)
            held_keys.remove(key)

    for key in target_keys:
        if key not in held_keys:
            pydirectinput.keyDown(key)
            held_keys.add(key)


def press_enter():
    pydirectinput.keyDown("enter")
    time.sleep(0.5)
    pydirectinput.keyUp("enter")


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(run_name):
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("logs", f"{run_name}_{run_timestamp}")

    frames_dir = os.path.join(run_dir, "frames")
    progress_dir = os.path.join(run_dir, "progressbar_screenshots")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    csv_file = open(os.path.join(run_dir, "actions.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["episode", "step", "move", "crouch", "shoot", "dash", "hp", "damage_reward", "time"]
    )

    return frames_dir, progress_dir, csv_file, csv_writer


# ============================================================
# EPISODE LOGIC
# ============================================================

def run_episode(
    episode,
    cfg,
    sct,
    monitor,
    templates,
    first_left_region,
    first_right_region,
    second_left_region,
    front_region,
    card_front_template,
    frames_dir,
    progress_dir,
    csv_writer,
    hp_region,
    progress_region
):

    damage_tracker_instance = damage_tracker.DamageTracker(
        first_left_region,
        first_right_region,
        second_left_region,
        front_region,
        card_front_template
    )

    STEP_DELAY = cfg["step_delay"]
    FRAME_SAVE_INTERVAL = cfg["frame_save_interval"]

    DAMAGE_COOLDOWN = 1.0
    EPISODE_END_WAIT = 4.0
    EPISODE_START_WAIT = 3.6

    step = 0
    last_hp = 4
    current_hp = 4
    last_damage_time = 0.0

    while True:
        # -------- OBSERVATION --------
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img_small = cv2.resize(img_gray, (320, 180))

        display = cv2.resize(img_small, (640, 360), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Agent View", display)

        # -------- HP DETECTION --------
        hp_screenshot = sct.grab(hp_region)
        hp_img = np.array(hp_screenshot)
        hp_gray = cv2.cvtColor(hp_img, cv2.COLOR_BGRA2GRAY)

        detected_hp, score = detect_hp_from_template(hp_gray, templates)
        now = time.time()

        if detected_hp is not None and detected_hp < last_hp:
            if now - last_damage_time > DAMAGE_COOLDOWN:
                current_hp = last_hp - 1
                last_damage_time = now
            else:
                current_hp = last_hp
        else:
            current_hp = last_hp

        # -------- DAMAGE TRACKING --------
        damage_reward, ult_ready = damage_tracker_instance.update(sct)
        damage_reward = damage_reward or 0.0

        if damage_reward > 0:
            print(f"Damage reward: {round(damage_reward, 4)}")

        if ult_ready:
            print("ULT READY → pressing super")
            pydirectinput.press("d")
            damage_tracker_instance.reset()

        # -------- DEATH --------
        if current_hp == 0:
            release_all_keys()
            time.sleep(EPISODE_END_WAIT)

            progress_screenshot = sct.grab(progress_region)
            progress_img = np.array(progress_screenshot)
            progress_gray = cv2.cvtColor(progress_img, cv2.COLOR_BGRA2GRAY)

            progress_path = os.path.join(
                progress_dir, f"episode_{episode}_progress.png"
            )
            cv2.imwrite(progress_path, progress_gray)

            press_enter()
            time.sleep(EPISODE_START_WAIT)
            break

        last_hp = current_hp

        # -------- ACTION POLICY --------
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

        if dash:
            pydirectinput.press("shift")

        # -------- LOGGING --------
        if step % FRAME_SAVE_INTERVAL == 0:
            frame_path = os.path.join(
                frames_dir, f"ep{episode}_frame_{step:06d}.png"
            )
            cv2.imwrite(frame_path, img_small)

            csv_writer.writerow(
                [episode, step, move, crouch, shoot, dash, current_hp, damage_reward, time.time()]
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

        time.sleep(STEP_DELAY)
        step += 1


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = load_config()
    templates = load_hp_templates()

    hp_region = {"top": 990, "left": 80, "width": 120, "height": 60}
    progress_region = {"top": 100, "left": 500, "width": 900, "height": 600}

    # -------- CARD BAR COORDINATES --------

    first_left_region = {"top": 1008, "left": 163, "width": 4, "height": 44}
    first_right_region = {"top": 1008, "left": 183, "width": 4, "height": 44}
    second_left_region = {"top": 1008, "left": 190, "width": 4, "height": 44}

    # -------- FRONT TEMPLATE + REGION --------

    front_region = {"top": 1008, "left": 160, "width": 60, "height": 44}

    card_front_template = cv2.imread(
        "templates/card_front.png",
        cv2.IMREAD_GRAYSCALE
    )

    if card_front_template is None:
        raise ValueError("card_front.png not found")

    frames_dir, progress_dir, csv_file, csv_writer = setup_logging(cfg["run_name"])

    print("Switch to Cuphead in 5 seconds...")
    time.sleep(5)

    print("Starting first episode (initial Enter)...")
    press_enter()
    time.sleep(3.6)

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        try:
            for episode in range(1, cfg["num_episodes"] + 1):
                print(f"\n=== Episode {episode}/{cfg['num_episodes']} ===")

                run_episode(
                    episode,
                    cfg,
                    sct,
                    monitor,
                    templates,
                    first_left_region,
                    first_right_region,
                    second_left_region,
                    front_region,
                    card_front_template,
                    frames_dir,
                    progress_dir,
                    csv_writer,
                    hp_region,
                    progress_region
                )

            print("\nAll episodes finished.")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        finally:
            release_all_keys()
            csv_file.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
