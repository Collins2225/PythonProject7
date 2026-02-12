"""
Simple YOLOv8 Webcam Detection Test
Press 'q' to quit
"""

import cv2
import torch
from ultralytics import YOLO

# Fix for PyTorch 2.6+ compatibility
import torch.serialization

if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from ultralytics.nn.tasks import DetectionModel

        torch.serialization.add_safe_globals([DetectionModel])
    except:
        pass

print("=" * 60)
print("YOLOv8 Webcam Detection Test")
print("=" * 60)
print("\nControls:")
print("  - Press 'q' to quit")
print("  - Press 's' to save a screenshot")
print("\nStarting...\n")

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
print("✓ Model loaded successfully!\n")

# Try to open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" ERROR: Could not open webcam!")
    print("\nTroubleshooting:")
    print("  1. Check if another application is using the webcam")
    print("  2. Try a different camera index: cv2.VideoCapture(1)")
    print("  3. Check Windows Privacy Settings → Camera permissions")
    exit(1)

print("✓ Webcam opened successfully!")
print("\nDetection window will open. Press 'q' to quit.\n")

frame_count = 0
screenshot_count = 0

try:
    while True:
        # Read frame from webcam
        success, frame = cap.read()

        if not success:
            print(" Failed to read frame from webcam")
            break

        # Run YOLOv8 detection
        results = model(frame, conf=0.3, verbose=False)

        # Draw detections on frame
        annotated_frame = results[0].plot()

        # Add instructions overlay
        cv2.putText(annotated_frame, "Press 'q' to quit | 's' to save",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Count detections
        num_detections = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Objects detected: {num_detections}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"detection_screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"✓ Saved: {filename}")

        frame_count += 1

        # Print status every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

except KeyboardInterrupt:
    print("\n\nInterrupted by user (Ctrl+C)")

finally:
    # Cleanup
    print(f"\nTotal frames processed: {frame_count}")
    if screenshot_count > 0:
        print(f"Screenshots saved: {screenshot_count}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Cleanup complete. Goodbye!")