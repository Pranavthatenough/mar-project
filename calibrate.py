#!/usr/bin/env python3
"""
calibrate.py
------------
Standalone HSV calibration tool. Opens a live camera feed with trackbars
for H/S/V low/high bounds, shows the resulting mask side-by-side with the
raw frame, and can write the tuned range straight into arm_config.yaml.

Usage
-----
  python calibrate.py                  # calibrate "red" (range1) on camera 0
  python calibrate.py --color green
  python calibrate.py --color red --slot range2   # red's wraparound range
  python calibrate.py --source video.mp4
  python calibrate.py --focal-length   # measure focal length instead of HSV

Controls
--------
  s -> save the current trackbar values into arm_config.yaml
  q -> quit without saving
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "arm_config.yaml")


def open_camera(source):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: could not open camera/video source '{source}'")
        sys.exit(1)
    return cap


def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found.")
        sys.exit(1)
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_hsv_range(color, slot, lo, hi):
    """Write a calibrated HSV range back into arm_config.yaml, preserving formatting best-effort."""
    cfg = load_config()
    cfg.setdefault("detection", {}).setdefault(color, {})[slot] = {
        "h_lo": int(lo[0]), "h_hi": int(hi[0]),
        "s_lo": int(lo[1]), "s_hi": int(hi[1]),
        "v_lo": int(lo[2]), "v_hi": int(hi[2]),
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
    print(f"Saved detection.{color}.{slot} = "
          f"h[{lo[0]}-{hi[0]}] s[{lo[1]}-{hi[1]}] v[{lo[2]}-{hi[2]}] -> {CONFIG_PATH}")


def calibrate_hsv(source, color, slot):
    cap = open_camera(source)
    win = f"Calibrate: {color} ({slot})"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    defaults = {
        "red":   {"range1": ((0, 120, 70), (10, 255, 255)),
                  "range2": ((170, 120, 70), (180, 255, 255))},
        "green": {"range1": ((40, 70, 70), (85, 255, 255))},
    }
    lo0, hi0 = defaults.get(color, {}).get(slot, ((0, 100, 60), (180, 255, 255)))

    for name, val, maxv in [("H_lo", lo0[0], 180), ("H_hi", hi0[0], 180),
                             ("S_lo", lo0[1], 255), ("S_hi", hi0[1], 255),
                             ("V_lo", lo0[2], 255), ("V_hi", hi0[2], 255)]:
        cv2.createTrackbar(name, win, val, maxv, lambda x: None)

    print(f"\nCalibrating '{color}' / '{slot}'. Hold the ball in frame and "
          f"adjust the trackbars until only the ball is white in the mask.")
    print("  s = save to arm_config.yaml   |   q = quit without saving\n")

    lo = hi = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break
        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lo = np.array([cv2.getTrackbarPos("H_lo", win),
                        cv2.getTrackbarPos("S_lo", win),
                        cv2.getTrackbarPos("V_lo", win)])
        hi = np.array([cv2.getTrackbarPos("H_hi", win),
                        cv2.getTrackbarPos("S_hi", win),
                        cv2.getTrackbarPos("V_hi", win)])

        mask = cv2.inRange(hsv, lo, hi)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        combined = np.hstack([frame, masked])
        cv2.putText(combined, f"{color}/{slot}  H[{lo[0]}-{hi[0]}] S[{lo[1]}-{hi[1]}] V[{lo[2]}-{hi[2]}]",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_hsv_range(color, slot, lo, hi)

    cap.release()
    cv2.destroyAllWindows()


def calibrate_focal_length(source, known_distance_cm, known_diameter_cm):
    """
    Interactive focal-length calibration: hold the ball at a known distance,
    press 'c' to capture, and the tool computes focal_length_px from the
    measured pixel radius: f = r_px * distance / (diameter / 2).
    """
    cap = open_camera(source)
    win = "Focal Length Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"\nHold your calibration ball at exactly {known_distance_cm} cm from the "
          f"camera lens, centered in frame, then press 'c' to capture. Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        disp = frame.copy()
        h, w = disp.shape[:2]
        cv2.circle(disp, (w // 2, h // 2), 6, (0, 255, 255), -1)
        cv2.putText(disp, "Center ball on yellow dot, press 'c' to capture",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                        param1=100, param2=40, minRadius=10, maxRadius=300)
            if circles is None:
                print("No circle detected — try again with better lighting/contrast.")
                continue
            r_px = circles[0][0][2]
            focal_px = r_px * known_distance_cm / (known_diameter_cm / 2)
            print(f"Measured radius: {r_px:.1f}px  ->  focal_length_px = {focal_px:.1f}")
            cfg = load_config()
            cfg.setdefault("camera", {})["focal_length_px"] = round(float(focal_px), 1)
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
            print(f"Saved camera.focal_length_px -> {CONFIG_PATH}")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="HSV / focal-length calibration for ball_detector.py")
    p.add_argument("--source", default="0", help="Camera index or video file")
    p.add_argument("--color", default="red", choices=["red", "green"])
    p.add_argument("--slot", default="range1", choices=["range1", "range2"],
                    help="Which HSV range to tune (red needs both range1 and range2)")
    p.add_argument("--focal-length", action="store_true",
                    help="Run focal-length calibration instead of HSV tuning")
    p.add_argument("--distance-cm", type=float, default=30.0,
                    help="Known distance to the ball for focal-length calibration")
    p.add_argument("--diameter-cm", type=float, default=6.5,
                    help="Known ball diameter for focal-length calibration")
    args = p.parse_args()

    if args.focal_length:
        calibrate_focal_length(args.source, args.distance_cm, args.diameter_cm)
    else:
        calibrate_hsv(args.source, args.color, args.slot)


if __name__ == "__main__":
    main()
