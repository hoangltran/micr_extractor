"""Template matching engine for MICR E-13B character recognition."""

from pathlib import Path

import cv2
import numpy as np

from micr.engines.base import BaseMICREngine
from micr.models import CharacterResult

_DEFAULT_TEMPLATES_DIR = Path(__file__).parent.parent / "resources" / "templates"

# Standard height for MICR line normalization. All segmentation thresholds
# (min_area, min_h, grouping distances) are tuned for this resolution.
STANDARD_LINE_HEIGHT = 100


class TemplateMatchingEngine(BaseMICREngine):
    """
    Primary MICR recognition engine using structural feature matching.

    Works by:
    1. Segmenting individual characters via contour detection
    2. Grouping nearby contours for multi-part symbols
    3. Classifying each character using Hu moments + structural features
    """

    def __init__(self, templates_dir: Path | str | None = None):
        if templates_dir is None:
            templates_dir = _DEFAULT_TEMPLATES_DIR
        self.templates_dir = Path(templates_dir)
        self._ref_features: dict[str, dict] = {}
        self._load_reference_features()

    @property
    def name(self) -> str:
        return "template_matching"

    def _load_reference_features(self):
        """Load reference templates and compute their feature vectors."""
        char_names = [str(d) for d in range(10)] + [
            "transit", "on_us", "amount", "dash"
        ]
        for char_name in char_names:
            path = self.templates_dir / f"{char_name}.png"
            if path.exists():
                tpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if tpl is not None:
                    _, tpl_bin = cv2.threshold(tpl, 127, 255, cv2.THRESH_BINARY)
                    tpl_bin = _trim_roi(tpl_bin)
                    self._ref_features[char_name] = _compute_features(tpl_bin)

        if not self._ref_features:
            raise FileNotFoundError(
                f"No templates found in {self.templates_dir}. "
                "Run scripts/generate_templates.py first."
            )

    def recognize(self, micr_line_image: np.ndarray) -> list[CharacterResult]:
        """Segment and recognize all characters in the MICR line."""
        if len(micr_line_image.shape) == 3:
            micr_line_image = cv2.cvtColor(micr_line_image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(micr_line_image, 127, 255, cv2.THRESH_BINARY)

        # Ensure text=white on black background
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # Normalize to standard height so segmentation thresholds work
        # at any input resolution (e.g. 20px low-res or 300px high-res)
        normalized, scale_x, scale_y = _normalize_to_standard_height(binary)

        char_rois = self._segment_characters(normalized)

        results = []
        for roi, bbox in char_rois:
            best_char, best_score = self._classify_character(roi)
            # Map bbox back to original image coordinates
            ox = int(round(bbox[0] * scale_x))
            oy = int(round(bbox[1] * scale_y))
            ow = int(round(bbox[2] * scale_x))
            oh = int(round(bbox[3] * scale_y))
            results.append(
                CharacterResult(
                    character=best_char,
                    confidence=best_score,
                    bbox=(ox, oy, ow, oh),
                    engine=self.name,
                )
            )

        return results

    def _segment_characters(
        self, binary: np.ndarray
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """Segment individual characters using contour detection and grouping."""
        binary = self._mask_text_band(binary)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        h_img = binary.shape[0]
        min_h = h_img * 0.13  # Filter noise specks; Transit dots are ~0.16x height
        min_area = 50

        rects = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if h >= min_h and w >= 2 and area >= min_area:
                rects.append((x, y, w, h))

        if not rects:
            return []

        rects.sort(key=lambda r: r[0])

        # Fix noise-merged or noise-extended contours
        rects = self._fix_anomalous_contours(rects, binary)

        if not rects:
            return []

        # Group nearby contours belonging to the same character
        grouped = self._group_contours(rects, h_img)

        results = []
        for gx, gy, gw, gh in grouped:
            pad = 2
            x0 = max(0, gx - pad)
            y0 = max(0, gy - pad)
            x1 = min(binary.shape[1], gx + gw + pad)
            y1 = min(binary.shape[0], gy + gh + pad)
            roi = binary[y0:y1, x0:x1]
            if roi.size > 0:
                results.append((roi, (x0, y0, x1 - x0, y1 - y0)))

        return results

    @staticmethod
    def _fix_anomalous_contours(
        rects: list[tuple[int, int, int, int]], binary: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        Fix contours corrupted by edge noise.

        Handles two cases:
        1. Wide blobs: noise bridges two adjacent digits into one contour.
           Split at vertical projection gaps.
        2. Tall contours: noise at the image edge connects to a character,
           extending it vertically. Trim to the normal character zone.
        """
        if len(rects) < 3:
            return rects

        # Compute median dimensions from normal contours
        heights = sorted(r[3] for r in rects)
        widths = sorted(r[2] for r in rects)
        median_h = heights[len(heights) // 2]
        median_w = widths[len(widths) // 2]

        # Compute expected vertical character zone from normal contours
        normal_rects = [r for r in rects if 0.7 * median_h <= r[3] <= 1.3 * median_h]
        if normal_rects:
            char_top = int(np.median([r[1] for r in normal_rects]))
            char_bottom = int(np.median([r[1] + r[3] for r in normal_rects]))
        else:
            char_top = int(np.median([r[1] for r in rects]))
            char_bottom = int(np.median([r[1] + r[3] for r in rects]))

        h_img = binary.shape[0]
        fixed = []

        for x, y, w, h in rects:
            # Case 1: Wide blob — likely two merged characters
            if w > median_w * 1.6 and w > median_h * 0.9:
                sub_rects = _split_wide_contour(binary, x, y, w, h, median_w)
                if sub_rects:
                    fixed.extend(sub_rects)
                    continue

            # Case 2: Tall contour touching edge — noise-extended
            if h > median_h * 1.25 and (y < h_img * 0.15 or y + h > h_img * 0.85):
                margin = max(2, int(median_h * 0.05))
                new_y = max(0, char_top - margin)
                new_bottom = min(h_img, char_bottom + margin)
                new_h = new_bottom - new_y
                if new_h >= median_h * 0.5:
                    fixed.append((x, new_y, w, new_h))
                    continue

            fixed.append((x, y, w, h))

        return fixed

    @staticmethod
    def _mask_text_band(binary: np.ndarray) -> np.ndarray:
        """
        Mask out noise above and below the main MICR text band.

        Uses horizontal projection to find the vertical extent of the
        actual characters. Noise artifacts at the top/bottom edges of
        the image (scratches, marks) are zeroed out so they don't merge
        with real characters during contour detection.
        """
        h, w = binary.shape
        row_sums = np.sum(binary > 0, axis=1)
        max_sum = row_sums.max()

        if max_sum == 0:
            return binary

        # Rows with content above 10% of the peak are considered text rows.
        # Noise rows typically have <2% while character rows have >15%.
        threshold = max_sum * 0.10
        active = row_sums > threshold

        if not np.any(active):
            return binary

        active_indices = np.where(active)[0]
        top = int(active_indices[0])
        bottom = int(active_indices[-1])

        # Add a small margin (2% of height, min 2px)
        margin = max(2, int(h * 0.02))
        top = max(0, top - margin)
        bottom = min(h - 1, bottom + margin)

        # If the text band is already the full image, nothing to mask
        if top <= 0 and bottom >= h - 1:
            return binary

        result = binary.copy()
        if top > 0:
            result[:top, :] = 0
        if bottom < h - 1:
            result[bottom + 1:, :] = 0

        return result

    def _group_contours(
        self, rects: list[tuple[int, int, int, int]], img_height: int
    ) -> list[tuple[int, int, int, int]]:
        """
        Group nearby contours that belong to the same character.

        Uses a two-pass approach:
        Pass 1: Tight grouping for overlapping/very close contours.
        Pass 2: Merge isolated short contours with adjacent short groups
                 (handles Transit symbol: bar + dots are separate but close).
        """
        if not rects:
            return []

        tall_heights = [h for _, _, _, h in rects if h > img_height * 0.3]
        if tall_heights:
            median_h = sorted(tall_heights)[len(tall_heights) // 2]
        else:
            median_h = img_height * 0.5

        # Pass 1: tight grouping (overlapping or very close)
        tight_threshold = median_h * 0.20  # ~9px at standard resolution

        groups: list[list[tuple[int, int, int, int]]] = []
        current_group = [rects[0]]

        for i in range(1, len(rects)):
            prev_right = max(r[0] + r[2] for r in current_group)
            curr_left = rects[i][0]
            gap = curr_left - prev_right

            if gap < tight_threshold:
                current_group.append(rects[i])
            else:
                groups.append(current_group)
                current_group = [rects[i]]

        groups.append(current_group)

        # Pass 2: merge adjacent short groups (symbol parts)
        # A "short" group has max contour height < 0.75 * median char height.
        short_threshold = median_h * 0.75
        merge_distance = median_h * 0.30  # ~13px

        merged_groups = self._merge_short_groups(groups, short_threshold, merge_distance)

        # Convert groups to bounding boxes
        merged = []
        for group in merged_groups:
            x_min = min(r[0] for r in group)
            y_min = min(r[1] for r in group)
            x_max = max(r[0] + r[2] for r in group)
            y_max = max(r[1] + r[3] for r in group)
            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

        return merged

    @staticmethod
    def _merge_short_groups(
        groups: list[list[tuple[int, int, int, int]]],
        short_threshold: float,
        merge_distance: float,
    ) -> list[list[tuple[int, int, int, int]]]:
        """
        Merge isolated short groups with adjacent short groups.

        Transit symbol = tall bar (short contour) + two dots (short contours)
        that are close but not overlapping. This pass joins them.
        """

        def is_short(group: list[tuple[int, int, int, int]]) -> bool:
            return max(r[3] for r in group) < short_threshold

        def group_right(group: list[tuple[int, int, int, int]]) -> int:
            return max(r[0] + r[2] for r in group)

        def group_left(group: list[tuple[int, int, int, int]]) -> int:
            return min(r[0] for r in group)

        result = []
        i = 0
        while i < len(groups):
            current = list(groups[i])

            # Only attempt merge if current group is short
            if is_short(current):
                # Try merging with next group(s) if they're also short and close
                while i + 1 < len(groups):
                    next_group = groups[i + 1]
                    gap = group_left(next_group) - group_right(current)

                    if gap < merge_distance and is_short(next_group):
                        current.extend(next_group)
                        i += 1
                    else:
                        break

            result.append(current)
            i += 1

        return result

    def _classify_character(
        self, char_roi: np.ndarray
    ) -> tuple[str, float]:
        """
        Classify a character ROI using Hu moments and structural features.
        """
        if char_roi.shape[0] < 3 or char_roi.shape[1] < 3:
            return "?", 0.0

        char_roi = _trim_roi(char_roi)
        if char_roi.shape[0] < 3 or char_roi.shape[1] < 3:
            return "?", 0.0

        roi_features = _compute_features(char_roi)

        best_char = "?"
        best_score = 0.0

        for char_name, ref_features in self._ref_features.items():
            score = _compare_features(roi_features, ref_features)
            if score > best_score:
                best_score = score
                best_char = char_name

        return best_char, best_score


def _trim_roi(roi: np.ndarray) -> np.ndarray:
    """
    Trim low-density edges from a character ROI.

    Noise artifacts can add thin trailing pixels that inflate the bounding
    box. This trims columns/rows from the edges where the projection is
    below 10% of the peak, restoring the true character dimensions.
    """
    h, w = roi.shape[:2]
    if h < 3 or w < 3:
        return roi

    # Trim columns from left and right
    v_proj = np.sum(roi > 0, axis=0)
    v_peak = v_proj.max()
    if v_peak == 0:
        return roi

    v_threshold = v_peak * 0.10
    left = 0
    while left < w and v_proj[left] < v_threshold:
        left += 1
    right = w - 1
    while right > left and v_proj[right] < v_threshold:
        right -= 1

    # Trim rows from top and bottom
    h_proj = np.sum(roi > 0, axis=1)
    h_peak = h_proj.max()
    h_threshold = h_peak * 0.10
    top = 0
    while top < h and h_proj[top] < h_threshold:
        top += 1
    bottom = h - 1
    while bottom > top and h_proj[bottom] < h_threshold:
        bottom -= 1

    trimmed = roi[top:bottom + 1, left:right + 1]
    if trimmed.shape[0] < 3 or trimmed.shape[1] < 3:
        return roi
    return trimmed


def _normalize_to_standard_height(
    binary: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Resize a binary MICR line image to STANDARD_LINE_HEIGHT.

    Ensures segmentation thresholds work consistently regardless of input
    resolution. Images already near the standard height (80-120px) are
    returned as-is to avoid unnecessary resampling artifacts.

    Returns:
        (normalized_image, scale_x, scale_y) where scale factors map
        from normalized coordinates back to original coordinates.
    """
    h, w = binary.shape[:2]

    if 80 <= h <= 120:
        return binary, 1.0, 1.0

    scale = STANDARD_LINE_HEIGHT / h
    new_w = max(1, int(round(w * scale)))
    new_h = STANDARD_LINE_HEIGHT

    # INTER_NEAREST preserves sharp edges in binary images; smooth
    # interpolation (CUBIC/LANCZOS) blurs edges and corrupts characters
    # after re-thresholding.  INTER_AREA remains best for downscaling.
    interpolation = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(binary, (new_w, new_h), interpolation=interpolation)

    # Re-threshold to restore clean binary after interpolation
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    # Scale factors to map normalized coords → original coords
    scale_x = w / new_w
    scale_y = h / new_h

    return resized, scale_x, scale_y


def _split_wide_contour(
    binary: np.ndarray,
    x: int, y: int, w: int, h: int,
    median_w: float,
) -> list[tuple[int, int, int, int]] | None:
    """
    Split a wide merged contour at vertical projection gaps.

    When noise bridges two adjacent characters, the result is one contour
    spanning both. This function finds the gap between them using the
    vertical projection profile and returns separate bounding boxes.

    Returns None if no clean split point is found.
    """
    roi = binary[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    v_proj = np.sum(roi > 0, axis=0).astype(float)
    peak = v_proj.max()
    if peak == 0:
        return None

    # Look for a valley in the middle region (25%-75% of width)
    search_start = int(w * 0.25)
    search_end = int(w * 0.75)
    if search_end <= search_start:
        return None

    search_region = v_proj[search_start:search_end]
    if len(search_region) == 0:
        return None

    # Find the minimum in the search region
    min_idx = int(np.argmin(search_region))
    min_val = search_region[min_idx]

    # The valley must be significantly lower than the peaks (< 25% of peak)
    if min_val > peak * 0.25:
        return None

    split_col = search_start + min_idx

    # Find the extent of the gap (consecutive low columns)
    gap_threshold = peak * 0.20
    gap_start = split_col
    while gap_start > search_start and v_proj[gap_start - 1] < gap_threshold:
        gap_start -= 1
    gap_end = split_col
    while gap_end < search_end - 1 and v_proj[gap_end + 1] < gap_threshold:
        gap_end += 1

    # Split into two rectangles
    # Left part: original x to gap_start
    left_w = gap_start
    # Right part: gap_end to original end
    right_x = x + gap_end + 1
    right_w = w - gap_end - 1

    result = []
    if left_w > median_w * 0.3:
        result.append((x, y, left_w, h))
    if right_w > median_w * 0.3:
        result.append((right_x, y, right_w, h))

    return result if len(result) >= 2 else None


def _compute_features(binary_roi: np.ndarray) -> dict:
    """Compute a feature vector for a binary character image."""
    h, w = binary_roi.shape[:2]

    total_pixels = h * w
    white_pixels = np.sum(binary_roi > 0)
    density = white_pixels / max(total_pixels, 1)
    aspect_ratio = w / max(h, 1)

    # Hu moments (log-transformed)
    moments = cv2.moments(binary_roi)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_log = np.array([
        -np.sign(hm) * np.log10(max(abs(hm), 1e-30))
        for hm in hu_moments
    ])

    # Horizontal projection profile (normalized, resampled to 16 bins)
    h_proj = np.sum(binary_roi, axis=1).astype(float)
    if h_proj.max() > 0:
        h_proj /= h_proj.max()
    h_proj_fixed = cv2.resize(
        h_proj.reshape(-1, 1), (1, 16), interpolation=cv2.INTER_LINEAR
    ).flatten()

    # Vertical projection profile (normalized, resampled to 16 bins)
    v_proj = np.sum(binary_roi, axis=0).astype(float)
    if v_proj.max() > 0:
        v_proj /= v_proj.max()
    v_proj_fixed = cv2.resize(
        v_proj.reshape(-1, 1), (1, 16), interpolation=cv2.INTER_LINEAR
    ).flatten()

    # Number of connected components
    n_labels, _ = cv2.connectedComponents(binary_roi)
    n_components = n_labels - 1

    # Horizontal symmetry
    resized = cv2.resize(binary_roi, (32, 32))
    flipped_h = cv2.flip(resized, 1)
    h_sym = np.sum(resized == flipped_h) / (32 * 32)

    # Half densities
    mid_h = max(1, h // 2)
    mid_w = max(1, w // 2)
    top_density = np.sum(binary_roi[:mid_h, :] > 0) / max(mid_h * w, 1)
    bottom_density = np.sum(binary_roi[mid_h:, :] > 0) / max((h - mid_h) * w, 1)
    left_density = np.sum(binary_roi[:, :mid_w] > 0) / max(h * mid_w, 1)
    right_density = np.sum(binary_roi[:, mid_w:] > 0) / max(h * (w - mid_w), 1)

    return {
        "density": density,
        "aspect_ratio": aspect_ratio,
        "hu_moments": hu_log,
        "h_proj": h_proj_fixed,
        "v_proj": v_proj_fixed,
        "n_components": n_components,
        "h_symmetry": h_sym,
        "top_density": top_density,
        "bottom_density": bottom_density,
        "left_density": left_density,
        "right_density": right_density,
    }


def _compare_features(feat1: dict, feat2: dict) -> float:
    """
    Compare two feature vectors and return a similarity score [0, 1].
    """
    scores = []
    weights = []

    # Hu moments distance (most discriminative, first 4 moments)
    hu_dist = np.linalg.norm(feat1["hu_moments"][:4] - feat2["hu_moments"][:4])
    hu_score = max(0.0, 1.0 - hu_dist / 10.0)
    scores.append(hu_score)
    weights.append(3.0)

    # Density similarity
    density_diff = abs(feat1["density"] - feat2["density"])
    density_score = max(0.0, 1.0 - density_diff * 4.0)
    scores.append(density_score)
    weights.append(2.0)

    # Aspect ratio similarity
    ar_diff = abs(feat1["aspect_ratio"] - feat2["aspect_ratio"])
    ar_score = max(0.0, 1.0 - ar_diff * 3.0)
    scores.append(ar_score)
    weights.append(2.0)

    # Projection profile correlations
    with np.errstate(divide="ignore", invalid="ignore"):
        h_corr = np.corrcoef(feat1["h_proj"], feat2["h_proj"])[0, 1]
    if np.isnan(h_corr):
        h_corr = 0.0
    scores.append(max(0.0, (h_corr + 1.0) / 2.0))
    weights.append(1.5)

    with np.errstate(divide="ignore", invalid="ignore"):
        v_corr = np.corrcoef(feat1["v_proj"], feat2["v_proj"])[0, 1]
    if np.isnan(v_corr):
        v_corr = 0.0
    scores.append(max(0.0, (v_corr + 1.0) / 2.0))
    weights.append(1.5)

    # Component count similarity
    comp_diff = abs(feat1["n_components"] - feat2["n_components"])
    comp_score = max(0.0, 1.0 - comp_diff * 0.3)
    scores.append(comp_score)
    weights.append(1.0)

    # Symmetry similarity
    sym_diff = abs(feat1["h_symmetry"] - feat2["h_symmetry"])
    scores.append(max(0.0, 1.0 - sym_diff * 3.0))
    weights.append(0.5)

    # Half-density pattern similarity
    td = abs(feat1["top_density"] - feat2["top_density"])
    bd = abs(feat1["bottom_density"] - feat2["bottom_density"])
    ld = abs(feat1["left_density"] - feat2["left_density"])
    rd = abs(feat1["right_density"] - feat2["right_density"])
    half_score = max(0.0, 1.0 - (td + bd + ld + rd))
    scores.append(half_score)
    weights.append(1.0)

    total_weight = sum(weights)
    weighted_sum = sum(s * w for s, w in zip(scores, weights))

    return weighted_sum / total_weight
