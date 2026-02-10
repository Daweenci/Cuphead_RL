import time
import mss
import numpy as np
import cv2
import os

print("Starting in 5 seconds... switch to Cuphead!")
time.sleep(5)

# Create logs folder if not existing
os.makedirs("logs/debug", exist_ok=True)

# -------- FULL SCREEN --------
with mss.mss() as sct:
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)

img = np.array(screenshot)

# Grayscale + resize
img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
img_small = cv2.resize(img_gray, (320, 180))

# Save full observation
cv2.imwrite("logs/debug/full_observation_320x180.png", img_small)

# -------- HP REGION --------
hp_region = {"top": 1009, "left": 113, "width": 23, "height": 32}

with mss.mss() as sct:
    hp_screenshot = sct.grab(hp_region)

hp_img = np.array(hp_screenshot)

# Save raw HP crop (color)
cv2.imwrite("logs/debug/hp_raw_color.png", hp_img)

# Grayscale for analysis
hp_gray = cv2.cvtColor(hp_img, cv2.COLOR_BGRA2GRAY)
cv2.imwrite("logs/debug/hp_gray.png", hp_gray)

# Optional threshold version (helps a lot for OCR)
_, hp_thresh = cv2.threshold(hp_gray, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite("logs/debug/hp_threshold.png", hp_thresh)

print("Saved images to logs/debug/")
