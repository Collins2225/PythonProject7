"""
PyBullet Robot Simulation Environment
Creates a simple robot with camera in a room with objects
"""

import pybullet as p
import pybullet_data
import numpy as np
import cv2
from typing import Tuple, List
import time


class RobotSimulation:
    """PyBullet simulation for autonomous robot with camera"""

    def __init__(self, gui: bool = True):
        """
        Initialize PyBullet simulation

        Args:
            gui: Whether to show GUI (True) or run headless (False)
        """
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # Set up simulation parameters
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1. / 240.)

        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")

        # Create robot (simple box with camera)
        self.robot_id = self._create_robot()

        # Camera parameters
        self.camera_width = 640
        self.camera_height = 480
        self.fov = 60
        self.near_plane = 0.1
        self.far_plane = 10.0

        # Spawn objects in the environment
        self.object_ids = []
        self._spawn_objects()

        print("Simulation initialized successfully")

    def _create_robot(self) -> int:
        """Create a simple robot with a camera mount"""
        # Robot base (box)
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.1])
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.1],
                                          rgbaColor=[0.2, 0.2, 0.8, 1])

        robot_id = p.createMultiBody(baseMass=5.0,
                                     baseCollisionShapeIndex=base_collision,
                                     baseVisualShapeIndex=base_visual,
                                     basePosition=[0, 0, 0.1])

        return robot_id

    def _spawn_objects(self):
        """Spawn various objects in the environment"""
        # Object positions and types
        objects_to_spawn = [
            ("cube_small.urdf", [2, 0, 0.5], [1, 0, 0, 1]),  # Red cube
            ("sphere_small.urdf", [1.5, 1, 0.5], [0, 1, 0, 1]),  # Green sphere
            ("duck_vhacd.urdf", [2, -1, 0.3], [1, 1, 0, 1]),  # Yellow duck
            ("teddy_vhacd.urdf", [1, 1.5, 0.3], [0.6, 0.4, 0.2, 1]),  # Brown teddy
        ]

        # Add some walls to make it look like a room
        self._create_walls()

        for urdf_file, position, color in objects_to_spawn:
            try:
                obj_id = p.loadURDF(urdf_file, position, globalScaling=1.0)

                # Change color
                p.changeVisualShape(obj_id, -1, rgbaColor=color)
                self.object_ids.append(obj_id)
            except:
                print(f"Warning: Could not load {urdf_file}")

    def _create_walls(self):
        """Create simple walls for the room"""
        wall_height = 1.5
        wall_thickness = 0.1
        room_size = 5

        # Wall parameters: (half extents, position)
        walls = [
            ([room_size, wall_thickness, wall_height], [0, room_size, wall_height]),  # Back
            ([room_size, wall_thickness, wall_height], [0, -room_size, wall_height]),  # Front
            ([wall_thickness, room_size, wall_height], [room_size, 0, wall_height]),  # Right
            ([wall_thickness, room_size, wall_height], [-room_size, 0, wall_height]),  # Left
        ]

        for half_extents, position in walls:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                         rgbaColor=[0.7, 0.7, 0.7, 1])
            p.createMultiBody(baseMass=0,
                              baseCollisionShapeIndex=collision,
                              baseVisualShapeIndex=visual,
                              basePosition=position)

    def get_camera_image(self) -> np.ndarray:
        """
        Capture image from robot's camera

        Returns:
            RGB image as numpy array (height, width, 3)
        """
        # Get robot position and orientation
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)

        # Camera is mounted on top of robot, looking forward
        camera_pos = [robot_pos[0], robot_pos[1], robot_pos[2] + 0.3]

        # Convert quaternion to euler to get yaw
        euler = p.getEulerFromQuaternion(robot_orn)
        yaw = euler[2]

        # Calculate camera target (looking forward)
        target_distance = 3.0
        target_pos = [
            camera_pos[0] + target_distance * np.cos(yaw),
            camera_pos[1] + target_distance * np.sin(yaw),
            camera_pos[2]
        ]

        # Camera up vector
        up_vector = [0, 0, 1]

        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.camera_width / self.camera_height,
            nearVal=self.near_plane,
            farVal=self.far_plane
        )

        # Get camera image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert to numpy array (remove alpha channel)
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha

        return rgb_array

    def move_robot(self, linear_velocity: float, angular_velocity: float):
        """
        Move the robot with given velocities

        Args:
            linear_velocity: Forward velocity (m/s)
            angular_velocity: Rotational velocity (rad/s)
        """
        # Get current position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # Calculate new position
        dt = 1. / 240.
        new_x = pos[0] + linear_velocity * np.cos(yaw) * dt
        new_y = pos[1] + linear_velocity * np.sin(yaw) * dt
        new_yaw = yaw + angular_velocity * dt

        # Set new position and orientation
        new_orn = p.getQuaternionFromEuler([0, 0, new_yaw])
        p.resetBasePositionAndOrientation(self.robot_id,
                                          [new_x, new_y, pos[2]],
                                          new_orn)

    def step(self):
        """Step the simulation forward"""
        p.stepSimulation()

    def close(self):
        """Close the simulation"""
        p.disconnect()


if __name__ == "__main__":
    # Test the simulation
    print("Starting robot simulation test...")
    sim = RobotSimulation(gui=True)

    print("Simulation running. Robot will move in a circle.")
    print("Close the window to exit.")

    try:
        for _ in range(1000):
            # Move robot in a circle
            sim.move_robot(linear_velocity=0.5, angular_velocity=0.1)
            sim.step()

            # Capture and display camera image
            if _ % 30 == 0:  # Every 30 steps
                image = sim.get_camera_image()
                cv2.imshow("Robot Camera View", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            time.sleep(1. / 240.)
    except KeyboardInterrupt:
        print("Simulation interrupted")
    finally:
        cv2.destroyAllWindows()
        sim.close()