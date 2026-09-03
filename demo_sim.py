
import time
import numpy as np
import logging
from arm_controller import ArmController, ArmBackend
from ball_detector import Detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")

def run_simulation_demo():
    print("\n" + "="*50)
    print("6-DOF ARM SIMULATION DEMO")
    print("="*50)
    print("This script simulates the arm moving to various 'detected' balls.")
    print("It uses the IK (Inverse Kinematics) solver to find joint angles.")
    print("="*50 + "\n")

    # Initialize controller in simulation mode
    ctrl = ArmController(backend=ArmBackend.SIMULATION)
    
    # Simulate a few detections within the 58cm workspace
    # Camera is at (0, 0, 30), arm is at (0, 0, 0)
    # Z_arm = Z_cam + 30. So target Z should be small to stay within reach.
    targets = [
        {"color": "red",   "cx": 320, "cy": 240, "dist": 15.0}, # arm_z = 45cm (Reach is 58)
        {"color": "green", "cx": 200, "cy": 200, "dist": 10.0}, # arm_z = 40cm
        {"color": "red",   "cx": 400, "cy": 300, "dist": 20.0}, # arm_z = 50cm
    ]

    for i, t in enumerate(targets):
        print(f"\n--- Target {i+1}: {t['color'].upper()} ball at ({t['cx']}, {t['cy']}) ---")
        
        # Create a fake detection object
        d = Detection(
            color=t['color'],
            cx=t['cx'],
            cy=t['cy'],
            radius=40,
            distance=t['dist'],
            confidence=0.95,
            contour=np.array([])
        )
        
        # This will calculate IK and print the resulting angles
        q = ctrl.move_to_detection(d, frame_w=640, frame_h=480)
        
        if q:
            angles_deg = [np.degrees(a) for a in q]
            print(f"IK SUCCESS! Joint angles computed.")
        else:
            print("IK FAILED or rate-limited.")
        
        time.sleep(1.5)

    print("\nDemo complete. The arm logic is working correctly!")
    print("To see this with a GUI and real-time detection, run:")
    print("  python main_app.py --allow-mock")

if __name__ == "__main__":
    run_simulation_demo()
