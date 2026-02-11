"""Setup verification and quick start script"""

import sys
import subprocess


def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def main():
    print("=" * 60)
    print("Robot Detection Project - Setup Verification")
    print("=" * 60)

    # Check Python version
    print("\nPython version:", sys.version)
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        return
    else:
        print("Python version OK")

    # Check required packages
    required_packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'ultralytics': 'YOLOv8',
        'pybullet': 'PyBullet',
        'numpy': 'NumPy',
        'PIL': 'Pillow'
    }

    print("\nChecking required packages:")
    all_installed = True
    for package, name in required_packages.items():
        if check_package(package):
            print(f"{name} is installed")
        else:
            print(f"{name} is not installed")
            all_installed = False

    if not all_installed:
        print("\nSome packages are missing.")
        print("Run: pip install -r requirements.txt")
        return

    print("\n" + "=" * 60)
    print("All checks passed")
    print("=" * 60)

    print("\nYou are ready to proceed. Try the following commands:")
    print("\n1. Test YOLOv8 detection:")
    print("   cd src && python yolo_detector.py")
    print("\n2. Test robot simulation:")
    print("   cd src && python robot_sim.py")
    print("\n3. Run full demo:")
    print("   cd src && python integrated_demo.py")

    print("\n" + "=" * 60)

    # Download YOLOv8 model if needed
    try:
        from ultralytics import YOLO
        print("\nDownloading YOLOv8 model (first time only)...")
        model = YOLO('yolov8n.pt')
        print("YOLOv8 model ready")
    except Exception as e:
        print(f"Could not download model: {e}")
        print("The model will download automatically when you run the demo")


if __name__ == "__main__":
    main()
