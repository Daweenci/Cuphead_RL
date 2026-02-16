import cv2
import numpy as np
import time


class DamageTracker:
    def __init__(
        self,
        first_card_region,
        second_card_region,
        ult_region,
        ult_templates,
        ult_crops,
        num_cards=5,
        ult_threshold=0.8,
        reward_factor=1.0,
        front_match_threshold=0.72,
        ult_lock_duration=6.0,  # NEW
    ):

        self.num_cards = num_cards
        self.ult_threshold = ult_threshold
        self.reward_factor = reward_factor
        self.front_match_threshold = front_match_threshold
        self.ult_lock_duration = ult_lock_duration

        self.ult_lock_until = 0.0  # NEW

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

        if not (len(ult_templates) == 5 and len(ult_crops) == 5):
            raise ValueError("Provide exactly 5 templates and 5 crop configs.")

        self.ult_region = ult_region
        self.ult_templates = ult_templates
        self.ult_crops = ult_crops

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
    # Ult detection
    # ----------------------------------------------------------

    def _check_ult(self, sct):

        full_img = np.array(sct.grab(self.ult_region))
        gray_full = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_index = -1

        for idx, (template, crop) in enumerate(zip(self.ult_templates, self.ult_crops)):

            left = crop["left"]
            right = crop["right"]

            cropped_region = gray_full[:, left: gray_full.shape[1] - right]

            if cropped_region.shape[0] < template.shape[0] or \
            cropped_region.shape[1] < template.shape[1]:
                print(f"[ULT DEBUG] Template {idx} skipped (size mismatch)")
                continue

            result = cv2.matchTemplate(
                cropped_region,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            score = float(result.max())

            threshold = self.front_match_threshold
            delta = score - threshold
            percent_of_threshold = (score / threshold) * 100 if threshold > 0 else 0
            """
            print(
                f"[ULT DEBUG] Template {idx} "
                f"| score: {score:.4f} "
                f"| threshold: {threshold:.2f} "
                f"| delta: {delta:+.4f} "
                f"| % of threshold: {percent_of_threshold:.1f}%"
            )
            """
            if score > best_score:
                best_score = score
                best_index = idx
        """
        print(
            f"[ULT DEBUG] Best template: {best_index} "
            f"| score: {best_score:.4f} "
            f"| threshold: {self.front_match_threshold:.2f} "
            f"| delta: {best_score - self.front_match_threshold:+.4f}"
        )
        """
        if best_score >= self.front_match_threshold:
            print(f"[ULT DEBUG] >>> ULT DETECTED via template {best_index}")
            return True

        return False



    # ----------------------------------------------------------
    # MAIN UPDATE
    # ----------------------------------------------------------

    def update(self, sct):

        now = time.time()

        # -----------------------------
        # 1️⃣ LOCK PERIOD (NO REWARD)
        # -----------------------------
        if now < self.ult_lock_until:
            return 0.0, False

        total_reward = 0.0
        ult_ready = False

        full_img = np.array(sct.grab(self.full_region))

        for i in range(self.num_cards):

            base_x = i * self.card_spacing
            card_slice = full_img[:, base_x: base_x + self.bar_width]

            mask = self._fill_mask(card_slice)
            fill_ratio = self._compute_fill_ratio(mask)

            if i > 0:
                if self.stored_fill[i - 1] < self.ult_threshold:
                    continue

            if fill_ratio > self.stored_fill[i]:
                delta = fill_ratio - self.stored_fill[i]
                total_reward += delta * self.reward_factor
                self.stored_fill[i] = fill_ratio

        last_fill = self.stored_fill[-1]
        threshold = self.ult_threshold
        delta = last_fill - threshold
        percent_of_threshold = (last_fill / threshold) * 100 if threshold > 0 else 0
        """
        print(
            f"[FILL DEBUG] Last card fill: {last_fill:.4f} "
            f"| threshold: {threshold:.2f} "
            f"| delta: {delta:+.4f} "
            f"| % of threshold: {percent_of_threshold:.1f}%"
        )
        """
        if last_fill >= threshold:
            print("[FILL DEBUG] >>> Fill threshold reached, checking ult template")
            ult_ready = self._check_ult(sct)


        return total_reward, ult_ready

    # ----------------------------------------------------------

    def trigger_ult_lock(self):  # NEW
        self.ult_lock_until = time.time() + self.ult_lock_duration
        self.stored_fill = [0.0 for _ in range(self.num_cards)]

    def reset(self):
        self.stored_fill = [0.0 for _ in range(self.num_cards)]
