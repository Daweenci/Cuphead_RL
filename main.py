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
        5: ["templates/hp_5.png"],
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


def detect_hp_from_template(hp_gray, templates, min_confidence=0.7):
    best_score = -1
    best_hp = None

    for hp_value, template_list in templates.items():
        for tmpl in template_list:
            res = cv2.matchTemplate(hp_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            score = res.max()

            if score > best_score:
                best_score = score
                best_hp = hp_value

    # NEW: confidence check
    if best_score < min_confidence:
        return None, best_score

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

    csv_writer.writerow([
        "episode",
        "step",
        "move",
        "crouch",
        "shoot",
        "dash",
        "hp",
        "damage_reward",
        "damage_event",
        "ult_event",
        "time"
    ])

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
    first_card_region,
    second_card_region,
    ult_region,
    ult_templates,
    ult_crops,  
    frames_dir,
    progress_dir,
    csv_writer,
    hp_region,
    progress_region
):

    damage_tracker_instance = damage_tracker.DamageTracker(
        first_card_region,
        second_card_region,
        ult_region,
        ult_templates,
        ult_crops  
    )

    STEP_DELAY = cfg["step_delay"]
    FRAME_SAVE_INTERVAL = cfg["frame_save_interval"]

    DAMAGE_COOLDOWN = 1.0
    EPISODE_END_WAIT = 4.0
    EPISODE_START_WAIT = 3.6

    step = 0
    last_hp = 5
    current_hp = 5
    last_damage_time = 0.0

    while True:

        screenshot = sct.grab(monitor)
        img = np.array(screenshot)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img_small = cv2.resize(img_gray, (320, 180))

        display = cv2.resize(img_small, (640, 360), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Agent View", display)

        hp_screenshot = sct.grab(hp_region)
        hp_img = np.array(hp_screenshot)
        hp_gray = cv2.cvtColor(hp_img, cv2.COLOR_BGRA2GRAY)

        detected_hp, score = detect_hp_from_template(hp_gray, templates)
        now = time.time()

        if detected_hp is not None and detected_hp < last_hp:
            if now - last_damage_time > DAMAGE_COOLDOWN:
                current_hp = last_hp - 1
                last_damage_time = now

                print(
                    f"[HP LOSS] Step {step} | "
                    f"HP: {last_hp} → {current_hp} | "
                    f"Detected: {detected_hp} | "
                    f"Score: {score:.3f}"
                )

            else:
                current_hp = last_hp
        else:
            current_hp = last_hp

        damage_reward, ult_ready = damage_tracker_instance.update(sct)
        damage_reward = damage_reward or 0.0

        if damage_reward > 0:
            #print(f"[DAMAGE] Step {step} | Reward: {damage_reward:.4f}")
            print("")

        damage_event = damage_reward > 0
        ult_event = False

        if ult_ready:
            ult_event = True
            print(f"[EP {episode} | STEP {step}] ULT READY → PRESSING SUPER")
            pydirectinput.keyDown("d")
            time.sleep(0.5)
            pydirectinput.keyUp("d")
            damage_tracker_instance.trigger_ult_lock()

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

        if step % FRAME_SAVE_INTERVAL == 0:
            frame_path = os.path.join(
                frames_dir, f"ep{episode}_frame_{step:06d}.png"
            )
            cv2.imwrite(frame_path, img_small)

            csv_writer.writerow([
                episode,
                step,
                move,
                crouch,
                shoot,
                dash,
                current_hp,
                damage_reward,
                damage_event,
                ult_event,
                time.time()
            ])

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

    first_card_region = {"top": 1014, "left": 172, "width": 4, "height": 34}
    second_card_region = {"top": 1014, "left": 199, "width": 4, "height": 34}

    ult_region = {"top": 1012, "left": 274, "width": 40, "height": 40}

    ult_templates = [
        cv2.imread("templates/card_front_1.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("templates/card_front_2.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("templates/card_front_3.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("templates/card_front_4.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("templates/card_front_5.png", cv2.IMREAD_GRAYSCALE),
    ]

    ult_crops = [
        {"left": 0, "right": 0},
        {"left": 7, "right": 1},
        {"left": 8, "right": 3},
        {"left": 8, "right": 2},
        {"left": 4, "right": 0},
    ]

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
                    first_card_region,
                    second_card_region,
                    ult_region,
                    ult_templates,
                    ult_crops,
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
