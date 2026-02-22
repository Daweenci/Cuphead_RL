import time
import mss
import numpy as np
import cv2
import os

print("Starting in 5 seconds... switch to Cuphead!")
time.sleep(5)

# Create logs folder
os.makedirs("logs/debug", exist_ok=True)

def grab_gray(sct, region):
    """Grab region and convert BGRA -> Grayscale"""
    img = np.array(sct.grab(region))
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img, gray

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
    # HP REGION
    # ============================================================

    hp_region = {"top": 1009, "left": 113, "width": 23, "height": 32}

    hp_color, hp_gray = grab_gray(sct, hp_region)

    cv2.imwrite("logs/debug/hp_raw_color.png", hp_color)
    cv2.imwrite("logs/debug/hp_gray.png", hp_gray)

    _, hp_thresh = cv2.threshold(hp_gray, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite("logs/debug/hp_threshold.png", hp_thresh)

    # ============================================================
    # CARD REGIONS
    # ============================================================

    card1_region = {
        "top": 1014,
        "left": 172,
        "width": 4,
        "height": 31
    }

    card2_region = {
        "top": 1014,
        "left": 199,
        "width": 4,
        "height": 31
    }

    # Capture Card 1
    card1_color, card1_gray = grab_gray(sct, card1_region)
    cv2.imwrite("logs/debug/card1_raw_color.png", card1_color)
    cv2.imwrite("logs/debug/card1_gray.png", card1_gray)

    # Capture Card 2
    card2_color, card2_gray = grab_gray(sct, card2_region)
    cv2.imwrite("logs/debug/card2_raw_color.png", card2_color)
    cv2.imwrite("logs/debug/card2_gray.png", card2_gray)

    # Optional: whole card bar
    cards_bar_region = {"top": 1009, "left": 113, "width": 23, "height": 32}
    cards_bar_color, cards_bar_gray = grab_gray(sct, cards_bar_region)

    cv2.imwrite("logs/debug/cards_bar_raw_color.png", cards_bar_color)
    cv2.imwrite("logs/debug/cards_bar_gray.png", cards_bar_gray)

print("Saved all debug images to logs/debug/")