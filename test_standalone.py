"""
test_standalone.py
------------------
Self-contained demo — requires ONLY OpenCV and NumPy.
No ROS, no Gazebo, no hardware needed.

What it tests:
  1. BallDetector on a synthetic test frame (colored circles)
  2. IK solver: FK → target → IK → FK round-trip error
  3. ArmController in simulation mode

Run: python test_standalone.py
"""

import sys
import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("test")

PASS = "\033[92m✔\033[0m"
FAIL = "\033[91m✗\033[0m"


# ─── Test 1: Ball detector on synthetic frame ─────────────────────────────────

def make_test_frame() -> np.ndarray:
    """Create a 640x480 BGR frame with one red and one green circle.

    Radii are chosen so the resulting estimated distance (see
    ball_detector.FOCAL_LENGTH_PX / BALL_DIAMETER_CM) keeps both balls
    within the arm's reachable workspace once the camera's mount height
    (arm_config.yaml: camera.position_xyz_cm) is added on top in
    arm_controller.camera_to_arm_frame().
    """
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50   # dark gray bg
    # Red ball at (160, 240) radius 90px -> ~22cm estimated distance
    cv2.circle(frame, (160, 240), 90, (0, 30, 200), -1)
    # Green ball at (480, 240) radius 70px -> ~28cm estimated distance
    cv2.circle(frame, (480, 240), 70, (30, 200, 50), -1)
    return frame


def test_detector():
    from ball_detector import BallDetector
    det = BallDetector(target_color="both", min_radius=10)
    frame = make_test_frame()
    detections = det.detect(frame)

    found_colors = {d.color for d in detections}
    ok_red   = "red"   in found_colors
    ok_green = "green" in found_colors

    print(f"  {PASS if ok_red   else FAIL} Red ball detected")
    print(f"  {PASS if ok_green else FAIL} Green ball detected")

    for d in detections:
        print(f"    → {d.color}  cx={d.cx} cy={d.cy} r={d.radius}px  "
              f"dist≈{d.distance:.1f}cm  conf={d.confidence:.0%}")

    # Save annotated frame for visual inspection
    annotated = det.annotate(frame, detections)
    out_path = "/tmp/ball_test_output.png"
    cv2.imwrite(out_path, annotated)
    print(f"  Annotated frame saved → {out_path}")
    return ok_red and ok_green


# ─── Test 2: IK round-trip ────────────────────────────────────────────────────

def test_ik():
    from ik_solver import IKSolver, ArmError
    solver = IKSolver()

    targets = [
        (20, 0,  20, "forward-center"),
        (15, 10, 15, "right-offset"),
        (10, -5, 25, "left-high"),
        ( 0, 20, 10, "pure-lateral"),
    ]
    all_ok = True
    for tx, ty, tz, name in targets:
        try:
            q   = solver.solve((tx, ty, tz), position_only=True)
            pos = solver.fk_position(q)
            err = np.linalg.norm(np.array([tx, ty, tz]) - np.array(pos))
            ok  = err < 0.5   # 5mm tolerance
            all_ok = all_ok and ok
            deg = [f"{d:+.1f}°" for d in solver.angles_to_degrees(q)]
            print(f"  {PASS if ok else FAIL} {name:20s}  err={err:.3f}cm  "
                  f"angles={deg}")
        except ArmError as e:
            print(f"  {FAIL} {name:20s}  ArmError: {e}")
            all_ok = False

    # Test unreachable target
    try:
        solver.solve((999, 999, 999), position_only=True)
        print(f"  {FAIL} Unreachable target should have raised ArmError")
        all_ok = False
    except ArmError:
        print(f"  {PASS} Unreachable target correctly raises ArmError")

    return all_ok


# ─── Test 3: Arm controller (simulation) ─────────────────────────────────────

def test_arm_controller():
    from ball_detector import Detection
    from arm_controller import ArmController, ArmBackend
    import time

    ctrl = ArmController(backend=ArmBackend.SIMULATION)

    # Fake detection
    d = Detection(
        color="red", cx=320, cy=240, radius=50,
        distance=25.0, confidence=0.85,
        contour=np.zeros((1, 1, 2), dtype=np.int32),
    )
    q = ctrl.move_to_detection(d, frame_w=640, frame_h=480)
    ok_move = q is not None
    print(f"  {PASS if ok_move else FAIL} move_to_detection returned joint angles")

    # Low-confidence detection → should skip
    d_low = Detection(
        color="green", cx=100, cy=100, radius=20,
        distance=40.0, confidence=0.15,   # below threshold
        contour=np.zeros((1, 1, 2), dtype=np.int32),
    )
    q_low = ctrl.move_to_detection(d_low, 640, 480)
    ok_skip = q_low is None
    print(f"  {PASS if ok_skip else FAIL} Low-confidence detection correctly skipped")

    # Home
    ctrl.home()
    angles = ctrl.get_current_angles_deg()
    ok_home = all(a == 0.0 for a in angles)
    print(f"  {PASS if ok_home else FAIL} Home position all zeros")

    return ok_move and ok_skip and ok_home


# ─── Test 4: Full integration (no hardware) ───────────────────────────────────

def test_integration():
    """Detect on synthetic frame → IK → controller — end-to-end."""
    from ball_detector import BallDetector
    from arm_controller import ArmController, ArmBackend

    det  = BallDetector(target_color="both", min_radius=10)
    ctrl = ArmController(backend=ArmBackend.SIMULATION)
    frame = make_test_frame()
    detections = det.detect(frame)

    if not detections:
        print(f"  {FAIL} Integration: no balls detected")
        return False

    d   = detections[0]
    q   = ctrl.move_to_detection(d, 640, 480)
    ok  = q is not None
    print(f"  {PASS if ok else FAIL} Integration: {d.color} ball → "
          f"angles {[round(a,1) for a in ctrl.get_current_angles_deg()]}°")
    return ok


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    print("\n" + "="*55)
    print("  Ball Detection + 6-DOF Arm — Test Suite")
    print("="*55)

    print("\n[1] BallDetector (synthetic frame)")
    results["detector"] = test_detector()

    print("\n[2] IK Solver (round-trip accuracy)")
    results["ik"] = test_ik()

    print("\n[3] ArmController (simulation mode)")
    results["arm_ctrl"] = test_arm_controller()

    print("\n[4] Integration (detect → IK → controller)")
    results["integration"] = test_integration()

    print("\n" + "="*55)
    total   = len(results)
    passed  = sum(results.values())
    print(f"  Results: {passed}/{total} tests passed")
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL} {name}")
    print("="*55 + "\n")

    sys.exit(0 if all(results.values()) else 1)
