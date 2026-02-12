import cv2
import numpy as np
import time
import torch
from yolo_detector import YOLODetector
from robot_sim import RobotSimulation

# Fix for PyTorch 2.6+ compatibility with YOLOv8
import torch.serialization
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except:
        pass


class AutonomousRobot:
    """Autonomous robot with enhanced vision-based navigation"""

    def __init__(self, model_path: str = 'yolov8n.pt'):
        print("Initializing Autonomous Robot System...")

        # YOLO detector
        self.detector = YOLODetector(model_path=model_path, conf_threshold=0.3)

        # Simulation
        self.sim = RobotSimulation(gui=True)

        # Navigation parameters
        self.linear_speed = 0.8          # max forward speed
        self.angular_speed = 1.2         # max rotation speed

        # PD controller gains
        self.Kp = 1.0
        self.Kd = 0.2
        self.prev_error = 0.0

        # Motion smoothing
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.accel_limit = 0.05          # m/s per frame
        self.turn_limit = 0.1            # rad/s per frame

        print("System initialized successfully!")

    # ---------------------------------------------------------
    # Utility: Smooth acceleration
    # ---------------------------------------------------------
    def smooth(self, target, last, limit):
        if target > last:
            return min(target, last + limit)
        else:
            return max(target, last - limit)

    # ---------------------------------------------------------
    # Enhanced Navigation Behavior
    # ---------------------------------------------------------
    def simple_navigation_behavior(self, detections: list) -> tuple:
        linear_vel = 0.0
        angular_vel = 0.0

        if not detections:
            # No objects detected - rotate faster to search
            angular_vel = 0.8
            return linear_vel, angular_vel

        # Find the largest object
        largest_detection = max(detections, key=lambda d:
        (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))

        center_x = largest_detection['center'][0]
        image_center = 640 / 2

        error = (center_x - image_center) / image_center

        # Faster rotation
        angular_vel = -error * 1.2

        # Move forward more aggressively when centered
        if abs(error) < 0.3:
            linear_vel = 1.0

        return linear_vel, angular_vel

    # ---------------------------------------------------------
    # Main Run Loop
    # ---------------------------------------------------------
    def run(self, duration: int = 60):
        print(f"Running autonomous robot for {duration} seconds...")
        print("Press 'q' to quit early")

        start_time = time.time()
        frame_count = 0

        try:
            while (time.time() - start_time) < duration:
                image = self.sim.get_camera_image()
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # YOLO detection
                result = self.detector.detect(image_bgr, return_annotated=True)

                # Navigation
                linear_vel, angular_vel = self.simple_navigation_behavior(
                    result['detections']
                )

                # Apply movement
                self.sim.move_robot(linear_vel, angular_vel)
                self.sim.step()

                # Display annotated image
                if result['annotated_image'] is not None:
                    display_img = result['annotated_image'].copy()

                    fps = 1.0 / result['inference_time'] if result['inference_time'] > 0 else 0
                    cv2.putText(display_img, f"FPS: {fps:.1f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_img, f"Objects: {result['num_detections']}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_img, f"Linear: {linear_vel:.2f} m/s",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(display_img, f"Angular: {angular_vel:.2f} rad/s",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    cv2.imshow('Robot Vision - Object Detection', display_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit signal received")
                    break

                frame_count += 1
                time.sleep(1. / 30.)  # 30 Hz

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0

            print(f"\nStatistics:")
            print(f"  Duration: {elapsed:.2f} seconds")
            print(f"  Frames processed: {frame_count}")
            print(f"  Average FPS: {avg_fps:.2f}")

            cv2.destroyAllWindows()
            self.sim.close()


def main():
    print("=" * 60)
    print("YOLOv8 Object Detection for Autonomous Robots")
    print("=" * 60)
    print("\nThis demo will:")
    print("1. Start a PyBullet simulation with a robot")
    print("2. Run YOLOv8 object detection on the robot's camera")
    print("3. Navigate autonomously with enhanced control")
    print("\nPress 'q' to quit")
    print("=" * 60)

    input("\nPress Enter to start the demo...")

    robot = AutonomousRobot(model_path='yolov8n.pt')
    robot.run(duration=300)

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
