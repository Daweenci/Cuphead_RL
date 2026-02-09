import time
import mss
import numpy as np
import cv2

print("Starting in 5 seconds... switch to Cuphead!")
time.sleep(5)

with mss.mss() as sct:
    # Full screen for RL
    monitor = sct.monitors[1]  # full primary screen
    screenshot = sct.grab(monitor)

# Convert to numpy array
img = np.array(screenshot)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

# Resize to smaller dimensions (e.g., 160x90) for CNN input
img_small = cv2.resize(img_gray, (320, 180))

# Display for testing
# cv2.imshow("Small Grayscale Screenshot", img_small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# HP region for OCR (adjust after testing)
hp_region = {"top": 1000, "left": 30, "width": 120, "height": 60}  # example values

with mss.mss() as sct:
    hp_screenshot = sct.grab(hp_region)

# Convert to numpy array and grayscale for OCR
hp_img = np.array(hp_screenshot)
hp_gray = cv2.cvtColor(hp_img, cv2.COLOR_BGRA2GRAY)

cv2.imshow("HP Region", hp_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("finished") 