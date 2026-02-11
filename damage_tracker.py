import cv2
import numpy as np
import os


class DamageTracker:
    def __init__(
        self,
        card_fill_template,
        card_front_template,
        first_card_region,
        second_card_region,
        num_cards=5,
        match_method=cv2.TM_CCOEFF_NORMED,
        ult_threshold=0.95,
        growth_threshold=0.0,
        debug=False,
        debug_base_path=None,
    ):
        """
        card_fill_template: template for a filled card (used for growth signal)
        card_front_template: template for glowing super-ready card
        first_card_region / second_card_region:
            dict with keys: top, left, width, height
        """

        self.card_fill_template = card_fill_template
        self.card_front_template = card_front_template
        self.match_method = match_method
        self.ult_threshold = ult_threshold
        self.growth_threshold = growth_threshold
        self.num_cards = num_cards
        self.debug = debug
        self.debug_base_path = debug_base_path
        self.frame_index = 0

        # Compute spacing between cards
        dx = second_card_region["left"] - first_card_region["left"]

        self.card_regions = []
        for i in range(num_cards):
            region = {
                "top": first_card_region["top"],
                "left": first_card_region["left"] + i * dx,
                "width": first_card_region["width"],
                "height": first_card_region["height"],
            }
            self.card_regions.append(region)

        # Start below threshold
        self.stored_scores = [-1.0 for _ in range(num_cards)]

    # ==========================================================
    # Internal matching
    # ==========================================================

    def _match(self, gray_img, template):
        result = cv2.matchTemplate(gray_img, template, self.match_method)
        return float(result.max())

    # ==========================================================
    # Update per frame
    # ==========================================================

    def update(self, sct):
        """
        Call once per frame.

        Returns:
            reward (float)
            ult_ready (bool)
        """

        frame_folder = None

        if self.debug and self.debug_base_path:
            frame_folder = os.path.join(
                self.debug_base_path,
                f"frame_{self.frame_index:06d}"
            )
            os.makedirs(frame_folder, exist_ok=True)

        total_reward = 0.0
        ult_ready = False

        for i, region in enumerate(self.card_regions):

            screenshot = sct.grab(region)
            img = np.array(screenshot)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            if frame_folder:
                cv2.imwrite(
                    os.path.join(frame_folder, f"card_{i}.png"),
                    img
                )

            score = self._match(gray, self.card_fill_template)

            # =========================
            # Growth logic (fixed)
            # =========================

            # Ignore low scores completely
            if score < self.growth_threshold:
                continue

            # First activation (avoid huge artificial delta)
            if self.stored_scores[i] < self.growth_threshold:
                self.stored_scores[i] = score
                continue

            # Normal monotonic increase
            if score > self.stored_scores[i]:
                delta = score - self.stored_scores[i]
                total_reward += delta
                self.stored_scores[i] = score

        # =========================
        # Ult detection (last card only)
        # =========================

        last_index = self.num_cards - 1
        last_score = self.stored_scores[last_index]

        if last_score >= self.ult_threshold:

            region = self.card_regions[last_index]
            screenshot = sct.grab(region)
            img = np.array(screenshot)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            front_score = self._match(gray, self.card_front_template)

            if front_score >= self.ult_threshold:
                ult_ready = True

        # =========================
        # Debug CSV
        # =========================

        if frame_folder:
            csv_path = os.path.join(frame_folder, "data.csv")
            with open(csv_path, "w") as f:
                f.write("card_index,stored_score\n")
                for idx, stored in enumerate(self.stored_scores):
                    f.write(f"{idx},{stored}\n")

        self.frame_index += 1

        return total_reward, ult_ready

    # ==========================================================
    # Reset after ult
    # ==========================================================

    def reset(self):
        self.stored_scores = [-1.0 for _ in range(self.num_cards)]
