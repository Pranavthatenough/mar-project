"""
arm_controller.py
-----------------
Bridges ball detection → IK solver → physical or simulated arm.

Output backends:
  - "serial"     : sends servo angles over USB serial (common for DIY arms)
  - "ros"        : publishes std_msgs/Float64MultiArray to /arm/joint_commands
  - "simulation" : prints angles only (safe default for WSL/no hardware)
"""

import time
import logging
from typing import List, Optional, Tuple
from enum import Enum

import numpy as np

from ball_detector import Detection, CONFIG
from ik_solver import IKSolver, ArmError

logger = logging.getLogger(__name__)

def parse_dh_params(config):
    if not config or 'arm' not in config or 'dh_params' not in config['arm']:
        return None
    params = []
    for p in config['arm']['dh_params']:
        params.append({
            "d": float(p['d']),
            "a": float(p['a']),
            "alpha": np.radians(p['alpha_deg']),
            "theta_offset": np.radians(p.get('theta_offset_deg', 0.0))
        })
    return params

class ArmBackend(str, Enum):
    SERIAL     = "serial"
    ROS        = "ros"
    SIMULATION = "simulation"


class ArmController:
    def __init__(
        self,
        backend:          ArmBackend    = ArmBackend.SIMULATION,
        ik_solver:        Optional[IKSolver] = None,
        camera_to_base:   Optional[np.ndarray] = None,   # 4×4 extrinsic matrix
        serial_port:      str  = "/dev/ttyUSB0",
        serial_baud:      int  = 115200,
        ros_topic:        str  = "/arm/joint_commands",
        move_delay_s:     float = 0.05,    # min seconds between moves
        home_position:    Optional[List[float]] = None,  # joint angles (rad)
    ):
        self.backend   = backend
        
        # Load from config if available
        dh = parse_dh_params(CONFIG)
        limits = None
        if CONFIG and 'arm' in CONFIG and 'joint_limits' in CONFIG['arm']:
            limits = [tuple(l) for l in CONFIG['arm']['joint_limits']]
            
        self.ik = ik_solver or IKSolver(dh_params=dh, joint_limits=limits)
        self.delay     = move_delay_s
        self._last_cmd = 0.0
        
        if home_position is None and CONFIG and 'arm' in CONFIG and 'home' in CONFIG['arm']:
            home_position = [np.radians(a) for a in CONFIG['arm']['home']]
        self._current_q: List[float] = (home_position or [0.0] * 6)

        # Camera → base-frame transform
        if camera_to_base is None:
            self.T_cam_base = np.eye(4)
            if CONFIG and 'camera' in CONFIG and 'position_xyz_cm' in CONFIG['camera']:
                xyz = CONFIG['camera']['position_xyz_cm']
                self.T_cam_base[:3, 3] = xyz
            else:
                self.T_cam_base[2, 3] = 30.0   # Default 30 cm (matches arm_6dof.urdf mount height)
        else:
            self.T_cam_base = camera_to_base

        # Backend init
        self._serial_conn = None
        self._ros_pub     = None
        if backend == ArmBackend.SERIAL:
            self._init_serial(serial_port, serial_baud)
        elif backend == ArmBackend.ROS:
            self._init_ros(ros_topic)

        logger.info(f"ArmController ready  backend={backend.value}")

    # ── Backend initializers ──────────────────────────────────────────────────

    def _init_serial(self, port: str, baud: int):
        try:
            import serial
            self._serial_conn = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            logger.info(f"Serial port {port} @ {baud} baud opened.")
        except ImportError:
            logger.warning("pyserial not installed. Falling back to simulation.")
            self.backend = ArmBackend.SIMULATION
        except Exception as e:
            logger.warning(f"Cannot open serial {port}: {e}. Falling back to simulation.")
            self.backend = ArmBackend.SIMULATION

    def _init_ros(self, topic: str):
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            rclpy.init()
            self._ros_node = Node("ball_arm_controller")
            self._ros_pub  = self._ros_node.create_publisher(Float64MultiArray, topic, 10)
            logger.info(f"ROS2 publisher on {topic}")
        except ImportError:
            logger.warning("rclpy not found. Falling back to simulation.")
            self.backend = ArmBackend.SIMULATION

    # ── Coordinate transform ──────────────────────────────────────────────────

    def camera_to_arm_frame(self, cam_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Transform 3-D point from camera frame to arm base frame."""
        p_cam = np.array([*cam_xyz, 1.0])
        p_arm = self.T_cam_base @ p_cam
        return (float(p_arm[0]), float(p_arm[1]), float(p_arm[2]))

    # ── Safety checks ─────────────────────────────────────────────────────────

    def _rate_limit(self) -> bool:
        now = time.time()
        if now - self._last_cmd < self.delay:
            return False
        self._last_cmd = now
        return True

    @staticmethod
    def _detect_valid(det: Optional[Detection], min_confidence: float = 0.4) -> bool:
        if det is None:
            return False
        if det.confidence < min_confidence:
            logger.debug(f"Low confidence {det.confidence:.2f} — skipping move.")
            return False
        return True

    # ── Send command ──────────────────────────────────────────────────────────

    def _send_angles(self, angles_deg: List[float]):
        if self.backend == ArmBackend.SIMULATION:
            formatted = [f"{a:+7.2f}°" for a in angles_deg]
            logger.info(f"[SIM] Joints → {' '.join(formatted)}")
            return

        if self.backend == ArmBackend.SERIAL and self._serial_conn:
            # Protocol: "J1:angle J2:angle ... J6:angle\n"
            msg = " ".join(f"J{i+1}:{a:.2f}" for i, a in enumerate(angles_deg)) + "\n"
            self._serial_conn.write(msg.encode())
            logger.debug(f"Serial → {msg.strip()}")
            return

        if self.backend == ArmBackend.ROS and self._ros_pub:
            from std_msgs.msg import Float64MultiArray
            msg = Float64MultiArray()
            msg.data = [float(np.radians(a)) for a in angles_deg]
            self._ros_pub.publish(msg)
            logger.debug(f"ROS → {msg.data}")

    # ── Public API ────────────────────────────────────────────────────────────

    def move_to_detection(
        self,
        detection:       Detection,
        frame_w:         int,
        frame_h:         int,
        gripper_open:    bool = True,
        position_only:   bool = True,
    ) -> Optional[List[float]]:
        """
        Compute IK for the detected ball and send joint commands.

        Returns joint angles (rad) on success, None if skipped/failed.
        """
        if not self._detect_valid(detection):
            return None
        if not self._rate_limit():
            return None

        # 3-D position in camera frame
        cam_xyz = detection.to_3d(frame_w, frame_h)
        # Transform to arm base frame
        arm_xyz = self.camera_to_arm_frame(cam_xyz)

        logger.info(
            f"Ball [{detection.color}] camera_xyz={cam_xyz}  arm_xyz={arm_xyz}  "
            f"dist={detection.distance}cm  conf={detection.confidence:.0%}"
        )

        try:
            q = self.ik.solve(arm_xyz, q_init=self._current_q, position_only=position_only)
        except ArmError as e:
            logger.error(f"IK failed: {e}")
            return None

        # Compute J6 (gripper orientation) to point towards the ball
        # J6 rotates around Z-axis, so use atan2(Y, X) of the ball position
        ball_x, ball_y = arm_xyz[0], arm_xyz[1]
        j6_angle = np.arctan2(ball_y, ball_x)
        q[5] = np.clip(j6_angle, np.radians(-180), np.radians(180))

        angles_deg = self.ik.angles_to_degrees(q)
        self._send_angles(angles_deg)
        self._current_q = q
        return q

    def compute_angles_for_detection(
        self,
        detection:       Detection,
        frame_w:         int,
        frame_h:         int,
        position_only:   bool = True,
    ) -> Optional[List[float]]:
        """Compute IK angles for the detected ball without sending commands."""
        if not self._detect_valid(detection):
            return None

        cam_xyz = detection.to_3d(frame_w, frame_h)
        arm_xyz = self.camera_to_arm_frame(cam_xyz)

        try:
            q = self.ik.solve(arm_xyz, q_init=self._current_q, position_only=position_only)
        except ArmError as e:
            logger.debug(f"IK compute failed: {e}")
            return None

        # Compute J6 (gripper orientation) to point towards the ball
        ball_x, ball_y = arm_xyz[0], arm_xyz[1]
        j6_angle = np.arctan2(ball_y, ball_x)
        q[5] = np.clip(j6_angle, np.radians(-180), np.radians(180))

        return q

    def home(self):
        """Move arm to home (all zeros)."""
        logger.info("Moving to home position.")
        self._send_angles([0.0] * 6)
        self._current_q = [0.0] * 6

    def stop(self):
        """Emergency stop — go home and close connections."""
        logger.warning("STOP called.")
        self.home()
        if self._serial_conn:
            self._serial_conn.close()

    def get_current_angles_deg(self) -> List[float]:
        return self.ik.angles_to_degrees(self._current_q)
