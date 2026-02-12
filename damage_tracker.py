import cv2
import numpy as np


class DamageTracker:
    def __init__(
        self,
        first_left_region,
        first_right_region,
        second_left_region,
        front_region,
        front_template,
        num_cards=5,
        ult_threshold=0.95,
        reward_factor=1.0,
        front_match_threshold=0.90,
    ):
        """
        first_left_region:
        first_right_region:
        second_left_region:
        front_region:
            dict with keys: top, left, width, height

        front_template:
            grayscale template of glowing ult-ready front

        ult_threshold:
            fill ratio threshold for last bar

        reward_factor:
            scaling factor for RL reward

        front_match_threshold:
            template match threshold for ult confirmation
        """

        self.num_cards = num_cards
        self.ult_threshold = ult_threshold
        self.reward_factor = reward_factor
        self.front_match_threshold = front_match_threshold

        self.front_region = front_region
        self.front_template = front_template

        # Abstand zwischen Karten berechnen
        dx = second_left_region["left"] - first_left_region["left"]

        # Balken-Regionen berechnen
        self.left_regions = []
        self.right_regions = []

        for i in range(num_cards):
            self.left_regions.append({
                "top": first_left_region["top"],
                "left": first_left_region["left"] + i * dx,
                "width": first_left_region["width"],
                "height": first_left_region["height"],
            })

            self.right_regions.append({
                "top": first_right_region["top"],
                "left": first_right_region["left"] + i * dx,
                "width": first_right_region["width"],
                "height": first_right_region["height"],
            })

        # gespeicherter Fortschritt pro Karte
        self.stored_fill = [0.0 for _ in range(num_cards)]

    # ==========================================================
    # Weiß-Maskierung (HSV)
    # ==========================================================

    def _white_mask(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)
        return mask

    # ==========================================================
    # Füllstand einer Balkenregion berechnen
    # ==========================================================

    def _compute_fill_ratio(self, mask):
        height, width = mask.shape
        filled_rows = 0

        # Von unten nach oben zählen
        for row in reversed(range(height)):
            white_pixels = np.count_nonzero(mask[row])

            if white_pixels >= width * 0.5:
                filled_rows += 1
            else:
                break

        return filled_rows / height

    # ==========================================================
    # Template Matching für Ult-Front
    # ==========================================================

    def _match_front(self, gray_img):
        result = cv2.matchTemplate(
            gray_img,
            self.front_template,
            cv2.TM_CCOEFF_NORMED
        )
        return float(result.max())

    # ==========================================================
    # Frame Update
    # ==========================================================

    def update(self, sct):
        total_reward = 0.0
        ult_ready = False

        # ----------------------------------------
        # 1) Balken-Fill berechnen
        # ----------------------------------------

        for i in range(self.num_cards):

            # Linker Balken
            left_img = np.array(sct.grab(self.left_regions[i]))
            left_mask = self._white_mask(left_img)
            left_fill = self._compute_fill_ratio(left_mask)

            # Rechter Balken
            right_img = np.array(sct.grab(self.right_regions[i]))
            right_mask = self._white_mask(right_img)
            right_fill = self._compute_fill_ratio(right_mask)

            # Sicherheitslogik: kleinerer Wert zählt
            fill_ratio = min(left_fill, right_fill)

            # Sequentielle Logik:
            # Karte i darf nur wachsen wenn Karte i-1 voll ist
            if i > 0:
                if self.stored_fill[i - 1] < self.ult_threshold:
                    continue

            # Monotonic clamp (kein Rückgang erlaubt)
            if fill_ratio > self.stored_fill[i]:
                delta = fill_ratio - self.stored_fill[i]
                total_reward += delta * self.reward_factor
                self.stored_fill[i] = fill_ratio

        # ----------------------------------------
        # 2) Ult Ready Check (letzte Karte + Front Template)
        # ----------------------------------------

        if self.stored_fill[-1] >= self.ult_threshold:

            front_img = np.array(sct.grab(self.front_region))
            front_gray = cv2.cvtColor(front_img, cv2.COLOR_BGR2GRAY)

            front_score = self._match_front(front_gray)

            if front_score >= self.front_match_threshold:
                ult_ready = True

        return total_reward, ult_ready

    # ==========================================================
    # Reset nach Ult
    # ==========================================================

    def reset(self):
        self.stored_fill = [0.0 for _ in range(self.num_cards)]
