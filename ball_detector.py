"""
ball_detector.py
----------------
Red/Green ball detection using HSV filtering + contour analysis.
Returns centroid (px), radius (px), estimated depth (cm), and confidence.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


import yaml
import os

# ─── Default HSV color ranges ────────────────────────────────────────────────
# Red wraps around hue=0/180, so we need two masks
DEFAULT_HSV_RANGES = {
    "red": [
        (np.array([0,   120,  70]),  np.array([10,  255, 255])),
        (np.array([170, 120,  70]),  np.array([180, 255, 255])),
    ],
    "green": [
        (np.array([40,  70,  70]),   np.array([85,  255, 255])),
    ],
}

# Known ball diameter in cm
DEFAULT_BALL_DIAMETER_CM = 6.5
# Focal length in pixels
DEFAULT_FOCAL_LENGTH_PX  = 600

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "arm_config.yaml")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None

CONFIG = load_config()

# Use config values or fall back to defaults
BALL_DIAMETER_CM = (CONFIG['ball']['real_diameter_cm'] if CONFIG and 'ball' in CONFIG 
                   else DEFAULT_BALL_DIAMETER_CM)
FOCAL_LENGTH_PX  = (CONFIG['camera']['focal_length_px'] if CONFIG and 'camera' in CONFIG 
                   else DEFAULT_FOCAL_LENGTH_PX)
MIN_CIRCULARITY  = (CONFIG['detection']['min_circularity'] if CONFIG and 'detection' in CONFIG 
                   else 0.10)

HSV_RANGES = DEFAULT_HSV_RANGES
if CONFIG and 'detection' in CONFIG:
    # Update HSV ranges from config if present
    for color in ["red", "green"]:
        if color in CONFIG['detection']:
            ranges = []
            c_cfg = CONFIG['detection'][color]
            for r_key in ["range1", "range2"]:
                if r_key in c_cfg:
                    r = c_cfg[r_key]
                    ranges.append((
                        np.array([r['h_lo'], r['s_lo'], r['v_lo']]),
                        np.array([r['h_hi'], r['s_hi'], r['v_hi']])
                    ))
            if ranges:
                HSV_RANGES[color] = ranges


@dataclass
class Detection:
    color:      str
    cx:         int           # centroid X (pixels)
    cy:         int           # centroid Y (pixels)
    radius:     int           # bounding circle radius (pixels)
    distance:   float         # estimated distance (cm)
    confidence: float         # 0..1 based on mask fill ratio
    contour:    np.ndarray    # raw contour for drawing
    timestamp:  float = field(default_factory=time.time)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.cx, self.cy)

    def to_3d(self, frame_w: int, frame_h: int, h_fov_deg: float = 60.0) -> Tuple[float, float, float]:
        """
        Convert pixel detection to approximate 3-D camera-frame coordinates (cm).
        Returns (X_right, Y_up, Z_forward).
        """
        fov_rad = np.radians(h_fov_deg)
        aspect  = frame_w / frame_h
        # angle per pixel
        ax = np.arctan(np.tan(fov_rad / 2) * 2 * (self.cx - frame_w / 2) / frame_w)
        ay = np.arctan(np.tan(fov_rad / 2 / aspect) * 2 * (frame_h / 2 - self.cy) / frame_h)
        Z  = self.distance
        X  = Z * np.tan(ax)
        Y  = Z * np.tan(ay)
        return (round(X, 1), round(Y, 1), round(Z, 1))


class BallDetector:
    """
    Detects red and/or green balls in a BGR frame.
    Supports tracking across frames with simple centroid matching.
    """

    def __init__(
        self,
        target_color: str = "both",    # "red" | "green" | "both"
        min_radius:   int = 15,         # px — ignore tiny blobs
        max_radius:   int = 250,        # px — ignore huge blobs
        morph_kernel: int = 5,
        smoothing_alpha: float = 0.4,   # EMA smoothing for jitter reduction
    ):
        assert target_color in ("red", "green", "both"), \
            "target_color must be 'red', 'green', or 'both'"
        self.target_color = target_color
        self.min_radius   = min_radius
        self.max_radius   = max_radius
        self.kernel       = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        self.alpha = smoothing_alpha
        self._tracked: dict = {}   # color -> Detection (latest smoothed)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, lo, hi)
        # morphological open (removes noise) then close (fills holes)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask

    @staticmethod
    def _estimate_distance(radius_px: int) -> float:
        if radius_px <= 0:
            return float("inf")
        return round(FOCAL_LENGTH_PX * (BALL_DIAMETER_CM / 2) / radius_px, 1)

    def _detect_color(self, hsv: np.ndarray, bgr: np.ndarray, color: str) -> List[Detection]:
        mask = self._build_mask(hsv, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * self.min_radius ** 2:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            radius = int(radius)
            if not (self.min_radius <= radius <= self.max_radius):
                continue
            # Circularity = 4*pi*area / perimeter^2 (1.0 = perfect circle).
            # Thresholded low (see arm_config.yaml: min_circularity) so that
            # near-circular blobs pass while thin lines/noise are rejected.
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            if circularity < MIN_CIRCULARITY:
                continue
            # confidence = ratio of mask pixels inside the bounding circle
            roi_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.circle(roi_mask, (int(cx), int(cy)), radius, 255, -1)
            fill = cv2.countNonZero(cv2.bitwise_and(mask, roi_mask))
            confidence = min(fill / (np.pi * radius ** 2), 1.0)
            distance   = self._estimate_distance(radius)
            results.append(Detection(
                color=color,
                cx=int(cx), cy=int(cy),
                radius=radius,
                distance=distance,
                confidence=round(confidence, 2),
                contour=cnt,
            ))
        # sort by area descending (largest first)
        results.sort(key=lambda d: d.radius, reverse=True)
        return results

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run detection on one BGR frame.
        Returns list of Detection objects, one per ball found.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        colors = (["red", "green"] if self.target_color == "both"
                  else [self.target_color])
        all_detections = []
        for color in colors:
            dets = self._detect_color(hsv, frame_bgr, color)
            all_detections.extend(dets)
        return all_detections

    def detect_best(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        """Return the single highest-confidence detection across all colors."""
        dets = self.detect(frame_bgr)
        if not dets:
            return None
        return max(dets, key=lambda d: d.confidence)

    def annotate(self, frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding circles, labels, and HUD onto a copy of frame."""
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        cv2.line(out, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), (200, 200, 200), 1)
        cv2.line(out, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), (200, 200, 200), 1)

        COLORS_BGR = {"red": (0, 0, 220), "green": (0, 200, 60)}

        for d in detections:
            clr = COLORS_BGR[d.color]
            # outer circle
            cv2.circle(out, d.center, d.radius, clr, 2)
            # centroid dot
            cv2.circle(out, d.center, 4, clr, -1)
            # label background
            label  = f"{d.color}  {d.distance:.0f}cm  {d.confidence*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx, ly = d.cx - tw // 2, d.cy - d.radius - 8
            cv2.rectangle(out, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4),
                          (30, 30, 30), -1)
            cv2.putText(out, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)
            # 3-D coords
            x3, y3, z3 = d.to_3d(w, h)
            coord_lbl = f"X:{x3:+.0f} Y:{y3:+.0f} Z:{z3:.0f} cm"
            cv2.putText(out, coord_lbl,
                        (d.cx - tw // 2, d.cy + d.radius + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, clr, 1, cv2.LINE_AA)

        # HUD
        status = f"Balls detected: {len(detections)}"
        cv2.putText(out, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        return out
