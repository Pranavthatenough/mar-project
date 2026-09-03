"""
launch_sim.py — ROS2 launch file
Starts: Gazebo + robot_state_publisher + joint_state_broadcaster + ball_detector node
"""
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, Command
import os

URDF_PATH = os.path.join(
    os.path.dirname(__file__), "arm_6dof.urdf"
)
CONTROLLERS_YAML = os.path.join(
    os.path.dirname(__file__), "config", "arm_controllers.yaml"
)


def load_robot_description() -> str:
    """
    Read arm_6dof.urdf and resolve the gz_ros2_control controller-config path.

    The URDF ships with a `$(find-pkg-share ball_arm_project)/config/...`
    placeholder so it stays portable if this project is later wrapped in a
    proper ROS2 package. Since this repo is *not* installed as a colcon
    package, we substitute that placeholder for a plain absolute path here
    instead, so `gz_ros2_control` can find its parameters file directly.
    """
    with open(URDF_PATH, "r") as f:
        urdf_xml = f.read()
    placeholder = "$(find-pkg-share ball_arm_project)/config/arm_controllers.yaml"
    return urdf_xml.replace(placeholder, CONTROLLERS_YAML)

def generate_launch_description():
    # Robot state publisher (broadcasts TF from URDF)
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{
            "robot_description": load_robot_description(),
            "use_sim_time": True,
        }],
    )

    # Gazebo Sim (Jazzy uses Gazebo Sim, formerly Ignition)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            ])
        ]),
        launch_arguments={"gz_args": "-r empty.sdf"}.items(),
    )

    # Spawn robot in Gazebo Sim (use the *resolved* description string so the
    # gz_ros2_control controller-parameters path is filled in, not the raw
    # file which still contains the placeholder)
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", "arm_6dof", "-string", load_robot_description(),
                   "-x", "0", "-y", "0", "-z", "0.05"],
        output="screen",
    )

    # Bridge for Camera (Gazebo Sim topic -> ROS 2 topic)
    # Jazzy uses image_bridge for more efficient image transport
    bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/image_raw"],
        output="screen",
    )

    # Bridge for Camera Info and other metadata
    info_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
    )

    # Joint state broadcaster
    jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    # Position controller
    pos_ctrl = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

    # Ball detector node (run directly as python node for simplicity)
    detector_node = Node(
        executable="python3",
        arguments=[os.path.join(os.path.dirname(__file__), "ball_detector_node.py")],
        name="ball_detector",
        parameters=[{
            "target_color": "both",
            "arm_enabled": True,
        }],
        output="screen",
    )

    return LaunchDescription([
        gazebo,
        robot_state_pub,
        bridge,
        info_bridge,
        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=4.0, actions=[jsb]),
        TimerAction(period=5.0, actions=[pos_ctrl]),
        TimerAction(period=3.0, actions=[detector_node]),
    ])
