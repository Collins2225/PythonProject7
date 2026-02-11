"""
YOLOv8 Object Detection Module
Handles model loading, inference, and visualization
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import time


class YOLODetector:
    """Wrapper class for YOLOv8 object detection"""

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.25):
        """
        Initialize YOLOv8 detector

        Args:
            model_path: Path to YOLOv8 model weights (yolov8n/s/m/l/x)
            conf_threshold: Confidence threshold for detections
        """
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names

    def detect(self, image: np.ndarray, return_annotated: bool = True) -> Dict:
        """
        Perform object detection on an image

        Args:
            image: Input image (BGR format)
            return_annotated: Whether to return annotated image

        Returns:
            Dictionary containing detections and annotated image
        """
        start_time = time.time()

        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)

        # Extract detection information
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.class_names[cls],
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                }
                detections.append(detection)

        inference_time = time.time() - start_time

        # Create annotated image if requested
        annotated_image = None
        if return_annotated and len(results) > 0:
            annotated_image = results[0].plot()

        return {
            'detections': detections,
            'annotated_image': annotated_image,
            'inference_time': inference_time,
            'num_detections': len(detections)
        }

    def detect_video(self, video_source: int = 0):
        """
        Run detection on video stream (webcam or video file)

        Args:
            video_source: Video source (0 for webcam, or path to video file)
        """
        cap = cv2.VideoCapture(video_source)

        print("Starting video detection. Press 'q' to quit.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Detect objects
            result = self.detect(frame)

            # Display results
            if result['annotated_image'] is not None:
                # Add FPS counter
                fps = 1.0 / result['inference_time'] if result['inference_time'] > 0 else 0
                cv2.putText(result['annotated_image'], f"FPS: {fps:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('YOLOv8 Detection', result['annotated_image'])

            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_object_position(self, detections: List[Dict], target_class: str) -> Tuple[int, int]:
        """
        Get the position of a specific object class

        Args:
            detections: List of detection dictionaries
            target_class: Target class name to find

        Returns:
            (x, y) center coordinates, or None if not found
        """
        for detection in detections:
            if detection['class_name'] == target_class:
                return tuple(detection['center'])
        return None


if __name__ == "__main__":
    # Example usage
    detector = YOLODetector(model_path='yolov8n.pt')

    # Test with webcam (if available)
    print("Testing with webcam (if available)...")
    print("Press 'q' to quit")
    detector.detect_video(0)