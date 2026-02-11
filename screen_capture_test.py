import time
import mss
import numpy as np
import cv2
import os

print("Starting in 5 seconds... switch to Cuphead!")
time.sleep(5)

# Create logs folder
os.makedirs("logs/debug", exist_ok=True)

with mss.mss() as sct:

    # ============================================================
    # FULL SCREEN (for reference)
    # ============================================================

    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img_small = cv2.resize(img_gray, (320, 180))

    cv2.imwrite("logs/debug/full_observation_320x180.png", img_small)

    # ============================================================
    # HP REGION (existing)
    # ============================================================

    hp_region = {"top": 1009, "left": 113, "width": 23, "height": 32}

    hp_screenshot = sct.grab(hp_region)
    hp_img = np.array(hp_screenshot)

    cv2.imwrite("logs/debug/hp_raw_color.png", hp_img)

    hp_gray = cv2.cvtColor(hp_img, cv2.COLOR_BGRA2GRAY)
    cv2.imwrite("logs/debug/hp_gray.png", hp_gray)

    _, hp_thresh = cv2.threshold(hp_gray, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite("logs/debug/hp_threshold.png", hp_thresh)

    # ============================================================
    # CARD REGIONS (NEW)
    # Adjust these values manually
    # ============================================================

    # First card (guess values)
    card1_region = {
        "top": 1008,
        "left": 163,
        "width": 24,
        "height": 44
    }

    # Second card (guess values)
    card2_region = {
        "top": 1008,
        "left": 190,
        "width": 24,
        "height": 44
    }

    # Optional: whole card bar (for overview)
    cards_bar_region = {
        "top": 1008,
        "left": 163,
        "width": 200,
        "height": 44
    }

    # Capture Card 1
    card1_img = np.array(sct.grab(card1_region))
    cv2.imwrite("logs/debug/card1_raw_color.png", card1_img)

    # Capture Card 2
    card2_img = np.array(sct.grab(card2_region))
    cv2.imwrite("logs/debug/card2_raw_color.png", card2_img)

    # Capture whole card bar (optional debug)
    cards_bar_img = np.array(sct.grab(cards_bar_region))
    cv2.imwrite("logs/debug/cards_bar_raw_color.png", cards_bar_img)

print("Saved all debug images to logs/debug/")
