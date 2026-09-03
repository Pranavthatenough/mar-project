
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from arm_controller import ArmController, ArmBackend
from ball_detector import Detection

def visualize_arm_3d():
    print("\n" + "="*50)
    print("6-DOF ARM 3D VISUALIZER")
    print("="*50)
    print("This will show you the physical structure of the 6-DOF arm.")
    print("="*50 + "\n")

    ctrl = ArmController(backend=ArmBackend.SIMULATION)
    
    # Target in front of the arm
    # Reach is ~58cm. Let's pick a very safe middle target.
    target_xyz = (20, 0, 30)
    
    try:
        # Solve IK for a target
        q = ctrl.ik.solve(target_xyz, q_init=[0, 0.5, -0.5, 0, 0, 0])
        
        # Get joint positions for plotting
        joint_positions = ctrl.ik.get_joint_positions(q)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract X, Y, Z coordinates
        xs = [p[0] for p in joint_positions]
        ys = [p[1] for p in joint_positions]
        zs = [p[2] for p in joint_positions]
        
        # Plot links
        ax.plot(xs, ys, zs, 'o-', linewidth=4, markersize=8, label='Arm Links (6-DOF)')
        
        # Plot joints (nodes)
        ax.scatter(xs, ys, zs, color='black', s=50)
        
        # Plot target
        ax.scatter([target_xyz[0]], [target_xyz[1]], [target_xyz[2]], 
                   color='red', s=150, label='Target Ball')
        
        # Draw a line from end-effector to target to show "reach"
        ax.plot([xs[-1], target_xyz[0]], [ys[-1], target_xyz[1]], [zs[-1], target_xyz[2]], 
                'r--', alpha=0.5)
        
        # Labels and limits
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_title('6-DOF Robotic Arm Structure')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = 60 # 60cm reach
        ax.set_xlim(-max_range/2, max_range)
        ax.set_ylim(-max_range/2, max_range/2)
        ax.set_zlim(0, max_range)
        
        print("Visualization window opening. Close it to continue.")
        plt.show()
        
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    visualize_arm_3d()
