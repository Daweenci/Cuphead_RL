import cv2
import numpy as np


class DamageTracker:
    def __init__(
        self,
        first_card_region,
        second_card_region,
        ult_region,          # large region matching card_front_1
        ult_templates,       # list of 5 grayscale templates
        ult_offsets,         # list of 5 {"left": x, "top": y}
        num_cards=5,
        ult_threshold=0.95,
        reward_factor=1.0,
        front_match_threshold=0.90,
    ):

        self.num_cards = num_cards
        self.ult_threshold = ult_threshold
        self.reward_factor = reward_factor
        self.front_match_threshold = front_match_threshold

        # --------------------------------------------------
        # Card bar geometry
        # --------------------------------------------------

        self.card_spacing = (
            second_card_region["left"] - first_card_region["left"]
        )

        self.bar_width = first_card_region["width"]
        self.bar_height = first_card_region["height"]
        self.top = first_card_region["top"]

        first_left = first_card_region["left"]
        last_right = first_left + (num_cards - 1) * self.card_spacing + self.bar_width

        self.full_region = {
            "top": self.top,
            "left": first_left,
            "width": last_right - first_left,
            "height": self.bar_height
        }

        self.stored_fill = [0.0 for _ in range(num_cards)]

        # --------------------------------------------------
        # Ult detection setup
        # --------------------------------------------------

        self.ult_region = ult_region
        self.ult_templates = ult_templates
        self.ult_offsets = ult_offsets

        if len(self.ult_templates) != 5 or len(self.ult_offsets) != 5:
            raise ValueError("Provide exactly 5 ult templates and 5 ult offsets.")

        for t in self.ult_templates:
            if t is None:
                raise ValueError("One or more ult templates failed to load.")

    # ----------------------------------------------------------
    # Orange fill mask
    # ----------------------------------------------------------

    def _fill_mask(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        lower = np.array([8, 135, 180])
        upper = np.array([26, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        # Vertical stabilization
        kernel = np.ones((3, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    # ----------------------------------------------------------

    def _compute_fill_ratio(self, mask):
        height, width = mask.shape
        filled_rows = 0

        for row in reversed(range(height)):
            white_pixels = np.count_nonzero(mask[row])
            if white_pixels >= width * 0.6:
                filled_rows += 1
            else:
                break

        return filled_rows / height

    # ----------------------------------------------------------
    # Offset-based ult detection
    # ----------------------------------------------------------

    def _check_ult(self, sct):

        full_img = np.array(sct.grab(self.ult_region))
        gray_full = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

        for template, offset in zip(self.ult_templates, self.ult_offsets):

            th, tw = template.shape

            x1 = offset["left"]
            y1 = offset["top"]
            x2 = x1 + tw
            y2 = y1 + th

            # Safety bounds check
            if x2 > gray_full.shape[1] or y2 > gray_full.shape[0]:
                continue

            cropped = gray_full[y1:y2, x1:x2]

            if cropped.shape != template.shape:
                continue

            result = cv2.matchTemplate(
                cropped,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            score = float(result.max())

            if score >= self.front_match_threshold:
                return True

        return False

    # ----------------------------------------------------------

    def update(self, sct):

        total_reward = 0.0
        ult_ready = False

        full_img = np.array(sct.grab(self.full_region))

        for i in range(self.num_cards):

            base_x = i * self.card_spacing

            card_slice = full_img[:, base_x: base_x + self.bar_width]

            mask = self._fill_mask(card_slice)
            fill_ratio = self._compute_fill_ratio(mask)

            # Sequential gating
            if i > 0:
                if self.stored_fill[i - 1] < self.ult_threshold:
                    continue

            # Monotonic increase only
            if fill_ratio > self.stored_fill[i]:
                delta = fill_ratio - self.stored_fill[i]
                total_reward += delta * self.reward_factor
                self.stored_fill[i] = fill_ratio

        # Ult detection
        if self.stored_fill[-1] >= self.ult_threshold:
            ult_ready = self._check_ult(sct)

        return total_reward, ult_ready

    # ----------------------------------------------------------

    def reset(self):
        self.stored_fill = [0.0 for _ in range(self.num_cards)]
