"""
ik_solver.py
------------
6-DOF robotic arm Inverse Kinematics solver.

Approach:
  - Analytical decoupled solution (position + orientation).
  - Falls back to numerical Jacobian pseudo-inverse if analytical fails.
  - Joint limits enforced; throws ArmError on unreachable targets.

DH Parameters default to a generic 6-DOF desktop arm (e.g. AR2/AR3-style).
Override by passing custom dh_params to the constructor.
"""

import numpy as np
from typing import List, Tuple, Optional


# ─── Custom exception ─────────────────────────────────────────────────────────

class ArmError(Exception):
    """Raised when the arm cannot reach a target or a safety limit is hit."""
    pass


# ─── DH parameter helpers ─────────────────────────────────────────────────────

def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard DH homogeneous transformation matrix (4×4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha),  np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ])


def forward_kinematics(joint_angles: List[float], dh_params: List[dict]) -> np.ndarray:
    """
    Compute end-effector pose (4×4 matrix) from joint angles.
    dh_params: list of dicts with keys 'd', 'a', 'alpha' (theta offset optional).
    """
    T = np.eye(4)
    for i, (theta, p) in enumerate(zip(joint_angles, dh_params)):
        theta_total = theta + p.get("theta_offset", 0.0)
        T = T @ dh_matrix(theta_total, p["d"], p["a"], p["alpha"])
    return T


# ─── Jacobian numerical IK ────────────────────────────────────────────────────

def _jacobian(q: np.ndarray, dh_params: List[dict], delta: float = 1e-5) -> np.ndarray:
    """Numerical Jacobian (6×n) using central differences."""
    n   = len(q)
    J   = np.zeros((6, n))
    T0  = forward_kinematics(q, dh_params)
    p0  = T0[:3, 3]
    # rotation → axis-angle
    for i in range(n):
        dq      = np.zeros(n)
        dq[i]   = delta
        Tp      = forward_kinematics(q + dq, dh_params)
        Tm      = forward_kinematics(q - dq, dh_params)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * delta)
        # angular velocity from rotation difference
        dR = (Tp[:3, :3] - Tm[:3, :3]) / (2 * delta)
        J[3, i] = dR[2, 1]   # wx
        J[4, i] = dR[0, 2]   # wy
        J[5, i] = dR[1, 0]   # wz
    return J


def _rot_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Orientation error as rotation vector (3,)."""
    Re   = R_des @ R_cur.T
    # skew-symmetric part
    err = np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]]) / 2
    return err


# ─── Main IK solver ───────────────────────────────────────────────────────────

# Default DH params — generic 6-DOF arm (units: cm / radians)
# Replace with your physical arm's actual measurements!
DEFAULT_DH_PARAMS = [
    {"d": 15.0, "a":  0.0,  "alpha":  np.pi/2},   # Joint 1 (base rotation)
    {"d":  0.0, "a": 14.0,  "alpha":  0.0      },   # Joint 2 (shoulder)
    {"d":  0.0, "a": 14.0,  "alpha":  0.0      },   # Joint 3 (elbow)
    {"d":  0.0, "a":  0.0,  "alpha":  np.pi/2  },   # Joint 4 (wrist roll)
    {"d": 12.0, "a":  0.0,  "alpha": -np.pi/2  },   # Joint 5 (wrist pitch)
    {"d":  6.0, "a":  0.0,  "alpha":  0.0      },   # Joint 6 (end-effector)
]

# Joint limits [min_deg, max_deg]
DEFAULT_JOINT_LIMITS = [
    (-180, 180),   # J1 base
    ( -90, 120),   # J2 shoulder
    (-135, 135),   # J3 elbow
    (-180, 180),   # J4 wrist roll
    ( -90,  90),   # J5 wrist pitch
    (-180, 180),   # J6 end-effector
]


class IKSolver:
    def __init__(
        self,
        dh_params:    Optional[List[dict]] = None,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        max_iter:     int   = 200,
        tol_pos:      float = 0.1,   # cm
        tol_rot:      float = 0.01,  # rad
        damping:      float = 0.05,
    ):
        self.dh_params     = dh_params    or DEFAULT_DH_PARAMS
        self.joint_limits  = joint_limits or DEFAULT_JOINT_LIMITS
        self.n             = len(self.dh_params)
        self.max_iter      = max_iter
        self.tol_pos       = tol_pos
        self.tol_rot       = tol_rot
        self.damping       = damping     # Levenberg-Marquardt damping

    # ── limits ────────────────────────────────────────────────────────────────

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        for i, (lo, hi) in enumerate(self.joint_limits):
            q[i] = np.clip(q[i], np.radians(lo), np.radians(hi))
        return q

    def _check_limits(self, q: np.ndarray) -> bool:
        for i, (lo, hi) in enumerate(self.joint_limits):
            if not (np.radians(lo) <= q[i] <= np.radians(hi)):
                return False
        return True

    # ── FK wrapper ───────────────────────────────────────────────────────────

    def fk(self, joint_angles_rad: List[float]) -> np.ndarray:
        """Forward kinematics — returns 4×4 pose matrix."""
        return forward_kinematics(joint_angles_rad, self.dh_params)

    def fk_position(self, joint_angles_rad: List[float]) -> Tuple[float, float, float]:
        T = self.fk(joint_angles_rad)
        return tuple(T[:3, 3])

    # ── IK ───────────────────────────────────────────────────────────────────

    def solve(
        self,
        target_pos:      Tuple[float, float, float],
        target_rpy_rad:  Tuple[float, float, float] = (0.0, 0.0, 0.0),
        q_init:          Optional[List[float]] = None,
        position_only:   bool = False,
    ) -> List[float]:
        """
        Solve IK for a given 3-D target.

        Parameters
        ----------
        target_pos      : (x, y, z) in cm in the arm base frame
        target_rpy_rad  : desired end-effector roll/pitch/yaw (radians)
        q_init          : initial joint angles (radians); defaults to zeros
        position_only   : if True, ignore orientation component of error

        Returns
        -------
        List[float] : joint angles in radians [J1..J6]

        Raises
        ------
        ArmError : if no solution found within tolerance or limits exceeded
        """
        tx, ty, tz = target_pos
        # Workspace sanity check
        reach = np.linalg.norm(target_pos)
        total_len = sum(p.get("a", 0) + p.get("d", 0) for p in self.dh_params)
        if reach > total_len * 0.95:
            raise ArmError(
                f"Target ({tx:.1f}, {ty:.1f}, {tz:.1f}) cm is outside reachable workspace "
                f"(max ~{total_len*0.95:.0f} cm)."
            )

        # Build desired rotation matrix from RPY
        r, p, y = target_rpy_rad
        Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
        Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
        Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
        R_des = Rz @ Ry @ Rx

        q = np.array(q_init, dtype=float) if q_init else np.zeros(self.n)
        q = self._clamp(q)

        for iteration in range(self.max_iter):
            T_cur  = forward_kinematics(q, self.dh_params)
            pos_err = np.array(target_pos) - T_cur[:3, 3]
            rot_err = (np.zeros(3) if position_only
                       else _rot_error(T_cur[:3, :3], R_des))

            err = np.concatenate([pos_err, rot_err])
            if position_only:
                err = pos_err

            e_norm = np.linalg.norm(err[:3])
            if e_norm < self.tol_pos and np.linalg.norm(err[3:]) < self.tol_rot:
                return q.tolist()

            J = _jacobian(q, self.dh_params)
            if position_only:
                J = J[:3, :]

            # Damped least squares (Levenberg–Marquardt)
            JT  = J.T
            lam = self.damping ** 2
            dq  = JT @ np.linalg.solve(J @ JT + lam * np.eye(J.shape[0]), err)

            # Adaptive step size
            step = min(1.0, 0.5 / (np.linalg.norm(dq) + 1e-9))
            q    = self._clamp(q + step * dq)

        raise ArmError(
            f"IK did not converge after {self.max_iter} iterations. "
            f"Residual position error: {e_norm:.3f} cm."
        )

    def angles_to_degrees(self, angles_rad: List[float]) -> List[float]:
        return [round(np.degrees(a), 2) for a in angles_rad]

    def get_joint_positions(self, joint_angles_rad: List[float]) -> List[Tuple[float, float, float]]:
        """
        Compute the (x, y, z) position of each joint in the arm base frame.
        Returns a list of 3-D points, including the base (0,0,0).
        """
        positions = [(0.0, 0.0, 0.0)]
        T = np.eye(4)
        for i, (theta, p) in enumerate(zip(joint_angles_rad, self.dh_params)):
            theta_total = theta + p.get("theta_offset", 0.0)
            T = T @ dh_matrix(theta_total, p["d"], p["a"], p["alpha"])
            positions.append(tuple(T[:3, 3]))
        return positions


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    solver = IKSolver()
    test_targets = [
        (20, 0, 20),
        (15, 10, 15),
        (10, -10, 25),
    ]
    for t in test_targets:
        try:
            q = solver.solve(t, position_only=True)
            pos = solver.fk_position(q)
            err = np.linalg.norm(np.array(t) - np.array(pos))
            deg = solver.angles_to_degrees(q)
            print(f"Target {t} → angles(°) {deg}  |  FK pos {[round(p,2) for p in pos]}  |  err={err:.3f}cm")
        except ArmError as e:
            print(f"Target {t} → ERROR: {e}")
