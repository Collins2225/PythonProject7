#!/usr/bin/env python3
"""
Automated Installation Script for Robot Detection Project
This script will install all required dependencies with proper error handling
"""

import subprocess
import sys
import os
import platform


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python 3.8 or higher is required!")
        print_info("Please upgrade Python and try again.")
        return False

    print_success("Python version is compatible")
    return True


def check_pip():
    """Check if pip is installed"""
    print_header("Checking pip")

    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"],
                                capture_output=True, text=True)
        print(result.stdout.strip())
        print_success("pip is installed")
        return True
    except Exception as e:
        print_error(f"pip is not installed: {e}")
        return False


def upgrade_pip():
    """Upgrade pip to latest version"""
    print_header("Upgrading pip")

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                       check=True)
        print_success("pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_warning(f"Could not upgrade pip: {e}")
        return False


def detect_cuda():
    """Detect if CUDA is available for GPU support"""
    print_header("Checking for GPU Support")

    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print_info("CUDA not available. Will use CPU (slower but works fine)")
            return False
    except ImportError:
        print_info("PyTorch not yet installed. Will check after installation.")
        return False


def install_pytorch():
    """Install PyTorch with appropriate configuration"""
    print_header("Installing PyTorch")

    system = platform.system()
    print(f"Detected OS: {system}")

    # PyTorch installation commands for different platforms
    if system == "Windows":
        print_info("Installing PyTorch for Windows (CPU version)...")
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision",
               "--index-url", "https://download.pytorch.org/whl/cpu"]
    elif system == "Darwin":  # macOS
        print_info("Installing PyTorch for macOS...")
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision"]
    else:  # Linux
        print_info("Installing PyTorch for Linux (CPU version)...")
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision",
               "--index-url", "https://download.pytorch.org/whl/cpu"]

    try:
        subprocess.run(cmd, check=True)
        print_success("PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install PyTorch: {e}")
        return False


def install_package(package_name, package_spec=None):
    """Install a single package"""
    if package_spec is None:
        package_spec = package_name

    print(f"Installing {package_name}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package_spec],
                       check=True, capture_output=True)
        print_success(f"{package_name} installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install {package_name}")
        return False


def install_requirements():
    """Install all requirements from requirements.txt"""
    print_header("Installing Required Packages")

    packages = [
        ("Ultralytics (YOLOv8)", "ultralytics==8.1.0"),
        ("OpenCV", "opencv-python>=4.8.0"),
        ("Pillow", "Pillow>=10.0.0"),
        ("PyBullet", "pybullet>=3.2.5"),
        ("NumPy", "numpy>=1.24.0"),
        ("Matplotlib", "matplotlib>=3.7.0"),
        ("tqdm", "tqdm>=4.65.0"),
        ("PyYAML", "pyyaml>=6.0"),
        ("SciPy", "scipy>=1.11.0"),
    ]

    failed_packages = []

    for name, spec in packages:
        if not install_package(name, spec):
            failed_packages.append(name)

    if failed_packages:
        print_warning(f"Some packages failed to install: {', '.join(failed_packages)}")
        return False

    print_success("All required packages installed successfully")
    return True


def install_optional_packages():
    """Install optional but recommended packages"""
    print_header("Installing Optional Packages")

    optional_packages = [
        ("Jupyter", "jupyter>=1.0.0"),
        ("TensorBoard", "tensorboard>=2.13.0"),
    ]

    print_info("Installing optional packages (recommended for development)...")

    for name, spec in optional_packages:
        install_package(name, spec)  # Don't fail if these don't install

    print_success("Optional packages installation complete")


def verify_installation():
    """Verify that all packages are properly installed"""
    print_header("Verifying Installation")

    packages_to_check = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'ultralytics': 'Ultralytics (YOLOv8)',
        'pybullet': 'PyBullet',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'scipy': 'SciPy',
        'tqdm': 'tqdm'
    }

    all_installed = True

    for module, name in packages_to_check.items():
        try:
            __import__(module)
            print_success(f"{name}")
        except ImportError:
            print_error(f"{name} - NOT FOUND")
            all_installed = False

    return all_installed


def download_yolo_model():
    """Download YOLOv8 model"""
    print_header("Downloading YOLOv8 Model")

    try:
        print_info("Downloading YOLOv8n model (first time only, ~6MB)...")
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print_success("YOLOv8 model downloaded successfully")
        return True
    except Exception as e:
        print_warning(f"Could not download model now: {e}")
        print_info("Model will be downloaded automatically when you first run the demo")
        return False


def create_directories():
    """Create necessary project directories"""
    print_header("Creating Project Directories")

    directories = [
        'src',
        'models',
        'data',
        'data/images',
        'data/images/train',
        'data/images/val',
        'data/labels',
        'data/labels/train',
        'data/labels/val',
        'notebooks',
        'docs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print_success("Project directories created")


def print_next_steps():
    """Print next steps for the user"""
    print_header("Installation Complete! 🎉")

    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}\n")

    print(f"{Colors.GREEN}1. Test YOLOv8 Detection:{Colors.ENDC}")
    print(f"   cd src")
    print(f"   python yolo_detector.py\n")

    print(f"{Colors.GREEN}2. Test Robot Simulation:{Colors.ENDC}")
    print(f"   cd src")
    print(f"   python robot_sim.py\n")

    print(f"{Colors.GREEN}3. Run Full Demo:{Colors.ENDC}")
    print(f"   cd src")
    print(f"   python integrated_demo.py\n")

    print(f"{Colors.GREEN}4. Learn Interactively:{Colors.ENDC}")
    print(f"   jupyter notebook notebooks/tutorial.ipynb\n")

    print(f"{Colors.YELLOW}Press 'q' to quit any demo{Colors.ENDC}\n")

    print(f"{Colors.BLUE}For detailed documentation, see:{Colors.ENDC}")
    print(f"   - README.md (comprehensive guide)")
    print(f"   - docs/QUICKSTART.md (quick start)")
    print(f"   - docs/ARCHITECTURE.md (technical details)\n")


def main():
    """Main installation process"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  Robot Detection Project - Automated Installation         ║")
    print("║  YOLOv8 + PyBullet for Autonomous Robots                  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")

    print_info("This script will install all required dependencies.")
    print_info(f"Installation directory: {os.getcwd()}\n")

    # Prompt user
    response = input("Continue with installation? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print_info("Installation cancelled.")
        return

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Check pip
    if not check_pip():
        print_error("Please install pip and try again.")
        sys.exit(1)

    # Step 3: Upgrade pip
    upgrade_pip()

    # Step 4: Install PyTorch first (it's large and specific)
    if not install_pytorch():
        print_warning("PyTorch installation had issues. Continuing anyway...")

    # Step 5: Install other requirements
    if not install_requirements():
        print_warning("Some packages failed to install. You may need to install them manually.")

    # Step 6: Install optional packages
    install_optional_packages()

    # Step 7: Verify installation
    if not verify_installation():
        print_warning("Some packages are missing. Please check the error messages above.")
        print_info("You can try installing missing packages manually with:")
        print_info("  pip install <package-name>")

    # Step 8: Create directories
    create_directories()

    # Step 9: Try to download YOLO model
    download_yolo_model()

    # Step 10: Check for GPU
    detect_cuda()

    # Step 11: Print next steps
    print_next_steps()

    print_success("Installation script completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nInstallation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error: {e}")
        sys.exit(1)