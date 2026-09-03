"""
main_app.py
-----------
Main application: Live camera + GUI control panel.

Usage
-----
  python main_app.py                         # webcam, simulation mode
  python main_app.py --source 0              # explicit webcam index
  python main_app.py --source video.mp4      # video file
  python main_app.py --color red             # detect red only
  python main_app.py --backend serial        # send to real arm
  python main_app.py --help
"""

import argparse
import logging
import threading
import time
import cv2
import numpy as np
import os
import platform
import sys

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ball_detector import BallDetector
from arm_controller import ArmController, ArmBackend

# ── Mock Camera ───────────────────────────────────────────────────────────────

class MockVideoCapture:
    """Simulates cv2.VideoCapture for testing without hardware."""
    def __init__(self):
        self.frame_idx = 0
        self.w, self.h = 640, 480
        # Create a moving ball simulation
        self.ball_x = 320
        self.ball_y = 240
        self.dx = 5
        self.dy = 3

    def isOpened(self): return True
    def release(self): pass
    def set(self, prop, val): pass
    def read(self):
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Background: dark gray
        frame[:] = (40, 40, 40)
        
        # Draw a fake red ball moving around
        self.ball_x += self.dx
        self.ball_y += self.dy
        if self.ball_x < 50 or self.ball_x > self.w - 50: self.dx *= -1
        if self.ball_y < 50 or self.ball_y > self.h - 50: self.dy *= -1
        
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), 30, (0, 0, 255), -1)
        
        # Add some noise
        noise = np.random.randint(0, 10, (self.h, self.w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        time.sleep(0.03) # simulate ~30fps
        return True, frame

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main_app")


# ─── HSV calibration helper ───────────────────────────────────────────────────

def hsv_calibration_window(cap: cv2.VideoCapture):
    """
    Optional tool: opens a trackbar window so you can tune HSV ranges live.
    Press 'q' to quit calibration.
    """
    cv2.namedWindow("HSV Calibration", cv2.WINDOW_NORMAL)
    params = [
        ("H_lo", 0, 180), ("H_hi", 180, 180),
        ("S_lo", 0, 255), ("S_hi", 255, 255),
        ("V_lo", 0, 255), ("V_hi", 255, 255),
    ]
    for name, val, maxv in params:
        cv2.createTrackbar(name, "HSV Calibration", val, maxv, lambda x: None)

    print("[Calibration] Adjust trackbars. Press 'q' to exit calibration.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 480))
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([cv2.getTrackbarPos("H_lo", "HSV Calibration"),
                        cv2.getTrackbarPos("S_lo", "HSV Calibration"),
                        cv2.getTrackbarPos("V_lo", "HSV Calibration")])
        hi = np.array([cv2.getTrackbarPos("H_hi", "HSV Calibration"),
                        cv2.getTrackbarPos("S_hi", "HSV Calibration"),
                        cv2.getTrackbarPos("V_hi", "HSV Calibration")])
        mask   = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("HSV Calibration", np.hstack([frame, result]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("HSV Calibration")
    print(f"[Calibration] Final range: lo={lo.tolist()} hi={hi.tolist()}")


# ─── Control Panel (tkinter) ──────────────────────────────────────────────────

class ControlPanel:
    """
    Tkinter GUI panel running in a separate thread.
    Provides:
      - Color selector (red / green / both)
      - Arm enable toggle
      - Emergency stop button
      - Live joint angle readout
      - Status label
    """

    def __init__(self, app: "App"):
        self.app   = app
        self.root  = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError:
            logger.warning("tkinter not available — control panel disabled.")
            return

        self.root = tk.Tk()
        self.root.title("Ball Detector + 6-DOF Arm")
        self.root.geometry("340x500")
        self.root.resizable(False, False)

        # ── Color selection ──
        frm_color = ttk.LabelFrame(self.root, text="Target Color")
        frm_color.pack(fill="x", padx=12, pady=8)
        self._color_var = tk.StringVar(value=self.app.color)
        for c in ("red", "green", "both"):
            ttk.Radiobutton(frm_color, text=c.capitalize(),
                            variable=self._color_var, value=c,
                            command=self._on_color_change).pack(side="left", padx=8)

        # ── Arm control ──
        frm_arm = ttk.LabelFrame(self.root, text="Arm Control")
        frm_arm.pack(fill="x", padx=12, pady=4)
        self._arm_var = tk.BooleanVar(value=self.app.arm_enabled)
        ttk.Checkbutton(frm_arm, text="Enable arm movement",
                        variable=self._arm_var,
                        command=self._on_arm_toggle).pack(anchor="w", padx=8, pady=4)

        btn_home = ttk.Button(frm_arm, text="Send to Home", command=self._on_home)
        btn_home.pack(fill="x", padx=8, pady=2)

        btn_stop = ttk.Button(frm_arm, text="⛔ EMERGENCY STOP",
                               command=self._on_stop)
        btn_stop.pack(fill="x", padx=8, pady=4)

        # ── Backend ──
        frm_bk = ttk.LabelFrame(self.root, text="Backend")
        frm_bk.pack(fill="x", padx=12, pady=4)
        self._bk_var = tk.StringVar(value=self.app.arm_ctrl.backend.value)
        for b in ("simulation", "serial", "ros"):
            ttk.Radiobutton(frm_bk, text=b, variable=self._bk_var,
                            value=b, command=self._on_backend_change).pack(side="left", padx=6)

        # ── Joint readout ──
        frm_joints = ttk.LabelFrame(self.root, text="Joint Angles (°)")
        frm_joints.pack(fill="x", padx=12, pady=4)
        self._joint_labels = []
        for i in range(6):
            row = ttk.Frame(frm_joints)
            row.pack(fill="x", padx=6, pady=1)
            ttk.Label(row, text=f"J{i+1}", width=3).pack(side="left")
            lbl = ttk.Label(row, text="  0.00°", width=12)
            lbl.pack(side="left")
            pb  = ttk.Progressbar(row, length=160, maximum=360, value=180)
            pb.pack(side="left", padx=4)
            self._joint_labels.append((lbl, pb))

        # ── Status ──
        frm_status = ttk.LabelFrame(self.root, text="Status")
        frm_status.pack(fill="x", padx=12, pady=4)
        self._status_lbl = ttk.Label(frm_status, text="Idle", foreground="gray")
        self._status_lbl.pack(padx=8, pady=4)

        # ── Camera Switcher ──
        frm_cam = ttk.LabelFrame(self.root, text="Camera Control")
        frm_cam.pack(fill="x", padx=12, pady=4)
        ttk.Button(frm_cam, text="Next Camera (Switch Index)", 
                   command=self._on_switch_camera).pack(fill="x", padx=8, pady=4)

        # ── Detection info ──
        frm_det = ttk.LabelFrame(self.root, text="Latest Detection")
        frm_det.pack(fill="x", padx=12, pady=4)
        self._det_lbl = ttk.Label(frm_det, text="—", justify="left")
        self._det_lbl.pack(padx=8, pady=4, anchor="w")

        # ── Calibration button ──
        ttk.Button(self.root, text="Open HSV Calibration",
                   command=self._on_calibrate).pack(fill="x", padx=12, pady=6)

        self._update_loop()
        self.root.mainloop()

    def _update_loop(self):
        if self.root is None:
            return
        try:
            # joints
            angles = self.app.arm_ctrl.get_current_angles_deg()
            for i, (lbl, pb) in enumerate(self._joint_labels):
                a = angles[i] if i < len(angles) else 0.0
                lbl.config(text=f"{a:+7.2f}°")
                pb["value"] = a + 180

            # status
            self._status_lbl.config(
                text=self.app.status_text,
                foreground="green" if "OK" in self.app.status_text else
                           "red"   if "ERROR" in self.app.status_text else "gray"
            )

            # latest detection
            d = self.app.latest_detection
            if d:
                self._det_lbl.config(
                    text=(f"Color: {d.color}  Conf: {d.confidence:.0%}\n"
                          f"Center: ({d.cx}, {d.cy}) px\n"
                          f"Distance: {d.distance:.1f} cm")
                )
            else:
                self._det_lbl.config(text="No ball detected")
        except Exception:
            pass
        self.root.after(200, self._update_loop)

    def _on_color_change(self):
        self.app.color = self._color_var.get()
        self.app.detector.target_color = self.app.color
        logger.info(f"Color → {self.app.color}")
        # Notify ROS node if active
        if self.app.arm_ctrl.backend == ArmBackend.ROS:
            from std_msgs.msg import String
            msg = String()
            msg.data = self.app.color
            if not hasattr(self.app.arm_ctrl, '_ros_color_pub'):
                self.app.arm_ctrl._ros_color_pub = self.app.arm_ctrl._ros_node.create_publisher(String, '/detector/set_color', 10)
            self.app.arm_ctrl._ros_color_pub.publish(msg)

    def _on_arm_toggle(self):
        self.app.arm_enabled = self._arm_var.get()
        logger.info(f"Arm enabled = {self.app.arm_enabled}")
        # Notify ROS node if active
        if self.app.arm_ctrl.backend == ArmBackend.ROS:
            from std_msgs.msg import Bool
            msg = Bool()
            msg.data = self.app.arm_enabled
            if not hasattr(self.app.arm_ctrl, '_ros_enable_pub'):
                self.app.arm_ctrl._ros_enable_pub = self.app.arm_ctrl._ros_node.create_publisher(Bool, '/detector/enable_arm', 10)
            self.app.arm_ctrl._ros_enable_pub.publish(msg)

    def _on_home(self):
        self.app.arm_ctrl.home()
        logger.info("Arm sent to home.")

    def _on_stop(self):
        logger.warning("Emergency stop triggered from GUI.")
        self.app.arm_ctrl.stop()
        self.app.running = False

    def _on_calibrate(self):
        hsv_calibration_window(self.app.cap)

    def _on_switch_camera(self):
        self.app.switch_camera()

    def _on_backend_change(self):
        bk = ArmBackend(self._bk_var.get())
        self.app.arm_ctrl.backend = bk
        logger.info(f"Backend → {bk.value}")


# ─── Main App ─────────────────────────────────────────────────────────────────

class App:
    def __init__(self, args):
        self.args            = args
        self.color           = args.color
        self.arm_enabled     = args.arm_enabled
        self.running         = True
        self.status_text     = "Initializing…"
        self.latest_detection = None

        # Video source
        self.is_ros = (args.backend == "ros")
        self.cap = None
        self._ros_frame = None

        if not self.is_ros:
            src = int(args.source) if args.source.isdigit() else args.source
            self._current_source_idx = src
            
            # Try to open the camera with multiple backends and indices if it fails
            self.cap = self._init_camera(src)
            
            if not self.cap or not self.cap.isOpened():
                logger.error(f"CRITICAL: Could not open any camera source.")
                if args.allow_mock:
                    logger.warning("Starting in MOCK CAMERA mode...")
                    self.cap = MockVideoCapture()
                else:
                    raise RuntimeError(f"CRITICAL: Could not open any camera source. "
                                     f"Checked source {src} and common fallbacks. "
                                     f"Try running with --allow-mock to test GUI/logic.")
            
            if hasattr(self.cap, 'set'):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            logger.info("ROS backend detected. Subscribing to /camera/image_raw...")

        # Detector
        self.detector = BallDetector(target_color=self.color)

        # Arm controller
        self.arm_ctrl = ArmController(backend=ArmBackend(args.backend))
        self.arm3d_enabled = HAS_MATPLOTLIB and self.arm_ctrl.backend == ArmBackend.SIMULATION
        if self.arm3d_enabled:
            self._init_arm_3d_plot()

        # If ROS, setup image subscriber
        if self.is_ros:
            try:
                from sensor_msgs.msg import Image
                from cv_bridge import CvBridge
                self.bridge = CvBridge()
                # Use the existing ROS node from arm_ctrl
                self.arm_ctrl._ros_node.create_subscription(
                    Image, '/camera/image_raw', self._on_ros_image, 10
                )
                logger.info("ROS image subscriber initialized.")
            except ImportError:
                logger.error("ROS dependencies (sensor_msgs, cv_bridge) not found!")

        # GUI (runs in background thread)
        try:
            self.panel = ControlPanel(self)
        except Exception as e:
            logger.warning(f"Control panel could not start: {e}")
            self.panel = None

        # Update is_ros based on actual backend (in case it fell back)
        self.is_ros = (self.arm_ctrl.backend == ArmBackend.ROS)

        # If fell back from ROS to simulation, init camera now
        if not self.is_ros and self.cap is None:
            src = int(args.source) if args.source.isdigit() else args.source
            self._current_source_idx = src
            self.cap = self._init_camera(src)
            if not self.cap or not self.cap.isOpened():
                logger.error(f"CRITICAL: Could not open any camera source.")
                if args.allow_mock:
                    logger.warning("Starting in MOCK CAMERA mode...")
                    self.cap = MockVideoCapture()
                else:
                    raise RuntimeError(f"CRITICAL: Could not open any camera source. "
                                     f"Checked source {src} and common fallbacks. "
                                     f"Try running with --allow-mock to test GUI/logic.")

        if args.calibrate and self.cap:
            hsv_calibration_window(self.cap)

    def _init_camera(self, primary_src):
        """Attempts to initialize camera with fallbacks for Windows and different indices."""
        backends = [None]
        is_wsl = False
        
        # Check for WSL
        if platform.system() == "Linux" and "microsoft" in platform.release().lower():
            is_wsl = True
            logger.warning("WSL detected. Note that USB cameras require 'usbipd-win' to be attached to WSL.")

        if platform.system() == "Windows":
            # CAP_DSHOW is often more reliable on Windows for many webcams
            backends.insert(0, cv2.CAP_DSHOW)
            # MSMF is another modern Windows backend
            backends.append(cv2.CAP_MSMF)

        # 1. Try primary source with all backends
        for backend in backends:
            try:
                logger.info(f"Trying camera source {primary_src} with backend {backend}...")
                cap = cv2.VideoCapture(primary_src, backend) if backend is not None else cv2.VideoCapture(primary_src)
                if cap.isOpened():
                    # Quick check if we can actually read a frame
                    ret, _ = cap.read()
                    if ret:
                        logger.info(f"Successfully opened camera {primary_src} (backend={backend})")
                        return cap
                    cap.release()
            except Exception as e:
                logger.debug(f"Failed to open source {primary_src} with backend {backend}: {e}")

        # 2. Fallback: Try other common indices (0, 1, 2, -1) if primary failed
        logger.warning(f"Failed to open primary source {primary_src}. Trying fallbacks...")
        for alt_idx in [0, 1, 2, -1]:
            if alt_idx == primary_src: continue # Skip what we already tried
            for backend in backends:
                try:
                    logger.info(f"Fallback: Trying camera source {alt_idx} with backend {backend}...")
                    cap = cv2.VideoCapture(alt_idx, backend) if backend is not None else cv2.VideoCapture(alt_idx)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            logger.info(f"Successfully opened fallback camera {alt_idx} (backend={backend})")
                            return cap
                        cap.release()
                except Exception:
                    continue
        
        if is_wsl:
            logger.error("\n" + "="*60 + 
                         "\nWSL CAMERA ACCESS ERROR:\n"
                         "WSL cannot access your Windows camera by default.\n"
                         "TO FIX THIS:\n"
                         "1. Run this app in Windows PowerShell instead of WSL.\n"
                         "2. OR: Use 'usbipd-win' to attach the USB camera to WSL.\n"
                         "   See: https://learn.microsoft.com/en-us/windows/wsl/connect-usb\n" +
                         "="*60)

        return None

    def _on_ros_image(self, msg):
        try:
            self._ros_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            logger.error(f"ROS image error: {e}")

    def _render_arm_simulation(self, joint_angles_rad):
        """Render a simple 2-D arm visualization for the simulation backend."""
        img_h, img_w = 480, 480
        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        origin = (80, img_h - 60)
        scale = 6.0

        # Draw reference grid lines
        for x in range(0, img_w, 80):
            cv2.line(canvas, (x, 0), (x, img_h), (20, 20, 20), 1)
        for y in range(0, img_h, 80):
            cv2.line(canvas, (0, y), (img_w, y), (20, 20, 20), 1)

        # Base anchor
        cv2.circle(canvas, origin, 6, (255, 255, 255), -1)
        cv2.putText(canvas, "BASE", (origin[0] - 30, origin[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if joint_angles_rad is None:
            joint_angles_rad = self.arm_ctrl._current_q

        if not hasattr(self.arm_ctrl, 'ik'):
            return canvas

        try:
            positions = self.arm_ctrl.ik.get_joint_positions(joint_angles_rad)
        except Exception:
            return canvas

        # Draw links in X-Z plane
        prev_px, prev_py = origin
        for i, pos in enumerate(positions[1:], start=1):
            px = int(origin[0] + pos[0] * scale)
            py = int(origin[1] - pos[2] * scale)
            cv2.line(canvas, (prev_px, prev_py), (px, py), (0, 180, 255), 4)
            cv2.circle(canvas, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(canvas, f"J{i}", (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            prev_px, prev_py = px, py

        # End effector marker
        cv2.circle(canvas, (prev_px, prev_py), 10, (0, 0, 255), 2)
        cv2.putText(canvas, "EE", (prev_px + 8, prev_py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(canvas, "ARM SIMULATION", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, "(X right, Z up)", (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return canvas

    def _init_arm_3d_plot(self):
        try:
            plt.ion()
            self.arm3d_fig = plt.figure(figsize=(5, 5))
            self.arm3d_ax = self.arm3d_fig.add_subplot(111, projection='3d')
            self.arm3d_line, = self.arm3d_ax.plot([], [], [], 'o-', color='#00ffff', linewidth=3, markersize=6)
            self.arm3d_target = self.arm3d_ax.scatter([], [], [], color='red', s=80)
            self.arm3d_ax.set_title('6-DOF Arm 3D Simulation')
            self.arm3d_ax.set_xlabel('X (cm)')
            self.arm3d_ax.set_ylabel('Y (cm)')
            self.arm3d_ax.set_zlabel('Z (cm)')
            self.arm3d_ax.set_xlim(-40, 40)
            self.arm3d_ax.set_ylim(-40, 40)
            self.arm3d_ax.set_zlim(0, 70)
            self.arm3d_ax.view_init(elev=20, azim=-60)
            self.arm3d_enabled = True
            self.arm3d_fig.canvas.draw()
            plt.pause(0.001)
        except Exception as e:
            logger.warning(f"Could not initialize 3D arm plot: {e}")
            self.arm3d_enabled = False

    def _update_arm_3d_plot(self, positions, target=None):
        if not getattr(self, 'arm3d_enabled', False):
            return
        try:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            zs = [p[2] for p in positions]
            self.arm3d_line.set_data(xs, ys)
            self.arm3d_line.set_3d_properties(zs)
            if target is not None:
                self.arm3d_target._offsets3d = ([target[0]], [target[1]], [target[2]])
            else:
                self.arm3d_target._offsets3d = ([], [], [])
            self.arm3d_ax.figure.canvas.draw_idle()
            plt.pause(0.001)
        except Exception as e:
            logger.debug(f"3D plot update failed: {e}")

    def switch_camera(self):
        """Cycle through camera indices (0 -> 1 -> 2 -> 0)."""
        if self.is_ros: return
        
        current_src = 0
        if hasattr(self, '_current_source_idx'):
            current_src = self._current_source_idx
        
        next_src = (current_src + 1) % 3
        logger.info(f"Switching camera to index {next_src}...")
        
        new_cap = self._init_camera(next_src)
        if new_cap and new_cap.isOpened():
            if self.cap: self.cap.release()
            self.cap = new_cap
            self._current_source_idx = next_src
            logger.info(f"Successfully switched to camera {next_src}")
        else:
            logger.warning(f"Could not open camera {next_src}, staying on {current_src}")

    def run(self):
        logger.info("Main loop started. Press 'q' in the video window to quit.")
        
        # Display check
        has_display = True
        if platform.system() == "Linux" and os.environ.get('DISPLAY') is None:
            logger.warning("No DISPLAY found. Running in HEADLESS mode (console only).")
            has_display = False

        if has_display:
            try:
                cv2.namedWindow("Ball Detection", cv2.WINDOW_NORMAL)
            except Exception as e:
                logger.warning(f"Could not create window: {e}. Switching to headless.")
                has_display = False

        fps_time = time.time()
        frame_count = 0

        # Initial placeholder frame for when waiting for ROS
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for camera...", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        while self.running:
            if self.is_ros:
                # Process ROS events
                import rclpy
                rclpy.spin_once(self.arm_ctrl._ros_node, timeout_sec=0.01)
                frame = self._ros_frame
                if frame is None:
                    if has_display:
                        cv2.imshow("Ball Detection", placeholder)
                        cv2.waitKey(1)
                    time.sleep(0.01)
                    continue
                ok = True
            else:
                ok, frame = self.cap.read()
                if not ok:
                    logger.warning("No frame received — attempting camera reconnect...")
                    if hasattr(self.cap, 'release'): self.cap.release()
                    time.sleep(1.0)
                    src = int(self.args.source) if self.args.source.isdigit() else self.args.source
                    self.cap = self._init_camera(src)
                    if not self.cap or not self.cap.isOpened():
                        self.status_text = "ERROR: Camera disconnected"
                        time.sleep(1.0)
                    continue

            frame_count += 1
            h, w = frame.shape[:2]

            # ── Detection ──────────────────────────────────────────────────
            detections = self.detector.detect(frame)
            
            # Smart ball selection: pick CLOSEST if multiple detected
            best_detection = None
            if detections:
                # Filter by confidence
                valid_dets = [d for d in detections if d.confidence >= 0.40]
                if valid_dets:
                    # Pick closest (minimum distance)
                    best_detection = min(valid_dets, key=lambda d: d.distance)
            
            self.latest_detection = best_detection

            # ── Arm move / IK compute ───────────────────────────────────────
            predicted_q = None
            action_status = "IDLE"
            
            if self.latest_detection:
                dist = self.latest_detection.distance
                
                # Distance-based filtering: ignore if too far or too close
                MAX_REACH = 50.0  # cm (arm workspace limit)
                MIN_REACH = 10.0  # cm (minimum safe distance)
                
                if dist > MAX_REACH:
                    action_status = f"OUT_OF_RANGE (too far: {dist:.1f}cm > {MAX_REACH}cm)"
                elif dist < MIN_REACH:
                    action_status = f"TOO_CLOSE ({dist:.1f}cm < {MIN_REACH}cm)"
                else:
                    action_status = "IN_RANGE"
                    if self.arm_enabled:
                        predicted_q = self.arm_ctrl.move_to_detection(self.latest_detection, w, h)
                    else:
                        predicted_q = self.arm_ctrl.compute_angles_for_detection(
                            self.latest_detection, w, h)

            # ── Status ─────────────────────────────────────────────────────
            if self.latest_detection:
                d = self.latest_detection
                self.status_text = (f"OK — {d.color.upper()} ball  {d.distance:.0f}cm  "
                                    f"conf={d.confidence:.0%}  [{action_status}]")
                
                # Smart console output
                angles = (self.arm_ctrl.ik.angles_to_degrees(predicted_q)
                          if predicted_q is not None else self.arm_ctrl.get_current_angles_deg())
                angles_str = " | ".join([f"J{i+1}:{a:+.1f}deg" for i, a in enumerate(angles)])
                
                arm_status = "MOVING" if predicted_q is not None else "IDLE"
                print(f"\r✓ [{d.color.upper()}] {d.distance:.1f}cm | Conf: {d.confidence:.0%} | Status: {action_status} | Arm: {arm_status} | {angles_str}", end="")
            elif detections:
                self.status_text = f"Low confidence detections filtered out"
            elif self.is_ros and self._ros_frame is None:
                self.status_text = "Waiting for ROS camera feed..."
            else:
                self.status_text = "Scanning... (no ball detected)"

            # ── Annotate + FPS ─────────────────────────────────────────────
            annotated = self.detector.annotate(frame, detections)
            
            # Add prominent ARM HUD if detection exists
            if self.latest_detection:
                angles = (self.arm_ctrl.ik.angles_to_degrees(predicted_q)
                          if predicted_q is not None else self.arm_ctrl.get_current_angles_deg())
                hud_y = 60
                hud_h = 30 + len(angles) * 20
                hud_w = 320
                top_left = (8, hud_y - 28)
                bottom_right = (top_left[0] + hud_w, top_left[1] + hud_h)

                # Semi-transparent background for readability
                overlay = annotated.copy()
                cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

                # Title with status
                status_color = (0, 255, 0) if action_status == "IN_RANGE" else (0, 165, 255)
                cv2.putText(annotated, f"ARM STATUS [{action_status}]:", (15, hud_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                for i, a in enumerate(angles):
                    text = f"J{i+1}: {a:+.2f}deg"
                    y = hud_y + 25 + (i * 20)
                    cv2.putText(annotated, text, (15, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(annotated, text, (15, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            now = time.time()
            fps = frame_count / (now - fps_time + 1e-9)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 100, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            if self.arm3d_enabled:
                q_for_plot = predicted_q if predicted_q is not None else self.arm_ctrl._current_q
                positions = self.arm_ctrl.ik.get_joint_positions(q_for_plot)
                target_pos = None
                if self.latest_detection:
                    target_pos = self.arm_ctrl.camera_to_arm_frame(self.latest_detection.to_3d(w, h))
                self._update_arm_3d_plot(positions, target=target_pos)

            if has_display:
                cv2.imshow("Ball Detection", annotated)
                if self.arm_ctrl.backend == ArmBackend.SIMULATION:
                    sim_img = self._render_arm_simulation(
                        predicted_q if predicted_q is not None else self.arm_ctrl._current_q
                    )
                    cv2.imshow("Arm Simulation", sim_img)
                    cv2.setWindowProperty("Arm Simulation", cv2.WND_PROP_TOPMOST, 1)
                cv2.setWindowProperty("Ball Detection", cv2.WND_PROP_TOPMOST, 1)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("h"):
                    self.arm_ctrl.home()
                elif key == ord("r"):
                    self.color = "red";  self.detector.target_color = "red"
                elif key == ord("g"):
                    self.color = "green"; self.detector.target_color = "green"
                elif key == ord("b"):
                    self.color = "both";  self.detector.target_color = "both"
                elif key == ord("a"):
                    self.arm_enabled = not self.arm_enabled
                    logger.info(f"Arm toggled: {self.arm_enabled}")
            else:
                # In headless mode, just log the status periodically
                if frame_count % 30 == 0:
                    logger.info(f"Headless Status: {self.status_text} | FPS: {fps:.1f}")
                time.sleep(0.01)

        self._cleanup()

    def _cleanup(self):
        logger.info("Shutting down.")
        self.running = False
        self.arm_ctrl.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Shutdown tkinter panel if it exists
        if hasattr(self, 'panel') and self.panel.root:
            try:
                self.panel.root.after(1, self.panel.root.destroy)
            except:
                pass

        if getattr(self, 'arm3d_enabled', False) and hasattr(self, 'arm3d_fig'):
            try:
                plt.close(self.arm3d_fig)
            except Exception:
                pass


# ─── CLI ──────────────────────────────────────────────────────────────────────

def interactive_startup_menu():
    """
    Interactive menu for user to select detection mode and arm enable.
    Returns: (color, arm_enabled, camera_source)
    """
    print("\n" + "="*60)
    print("  ROBOTIC ARM + BALL DETECTION SYSTEM")
    print("="*60)
    
    # Color selection
    print("\n🎯 SELECT DETECTION MODE:")
    print("  1 → RED BALLS ONLY")
    print("  2 → GREEN BALLS ONLY")
    print("  3 → BOTH RED & GREEN")
    color_choice = input("\nEnter choice (1-3) [default=3]: ").strip() or "3"
    color_map = {"1": "red", "2": "green", "3": "both"}
    color = color_map.get(color_choice, "both")
    
    # Arm enable
    print("\n🤖 ARM CONTROL:")
    arm_input = input("Enable arm movement? (y/n) [default=n]: ").strip().lower()
    arm_enabled = arm_input == "y"
    
    # Camera source
    print("\n📷 CAMERA SOURCE:")
    camera_source = input("Enter camera index or video file [default=0]: ").strip() or "0"
    
    print("\n" + "="*60)
    print(f"  MODE: {color.upper()} | ARM: {'ON' if arm_enabled else 'OFF'} | CAM: {camera_source}")
    print("="*60)
    print("\n⌨️ KEYBOARD SHORTCUTS:")
    print("  q → Quit")
    print("  r → Red only")
    print("  g → Green only")
    print("  b → Both colors")
    print("  a → Toggle arm on/off")
    print("  h → Home position")
    print("\n")
    
    return color, arm_enabled, camera_source


def parse_args():
    p = argparse.ArgumentParser(
        description="Red/Green Ball Detection + 6-DOF Arm Controller"
    )
    p.add_argument("--source",   default="0",
                   help="Camera index or video file path (default: 0)")
    p.add_argument("--color",    default="both",
                   choices=["red", "green", "both"],
                   help="Ball color to detect (default: both)")
    p.add_argument("--backend",  default="simulation",
                   choices=["simulation", "serial", "ros"],
                   help="Arm output backend (default: simulation)")
    p.add_argument("--arm-enabled", action="store_true", default=False,
                   help="Start with arm movement enabled")
    p.add_argument("--calibrate", action="store_true",
                   help="Open HSV calibration window before running")
    p.add_argument("--allow-mock", action="store_true", default=False,
                   help="Fallback to mock camera if hardware fails")
    p.add_argument("--interactive", action="store_true", default=False,
                   help="Show startup menu for user input")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Show interactive menu if --interactive flag or if running with no args
    if args.interactive or len(sys.argv) == 1:
        color, arm_enabled, source = interactive_startup_menu()
        args.color = color
        args.arm_enabled = arm_enabled
        args.source = source
    
    app  = App(args)
    app.run()
