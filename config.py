"""Central configuration for the CARLA ML driving project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# Camera (raw capture)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV = 90

# Model input after crop + resize (NVIDIA end-to-end style)
MODEL_INPUT_WIDTH = 200
MODEL_INPUT_HEIGHT = 66

# Data collection
DEFAULT_BEHAVIOR = "normal"   # careful / normal / aggressive
COLLECT_FRAMES = 5000
COLLECT_DELTA = 0.05

# Training
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Autonomous driving
TARGET_SPEED_KMH = 25.0
MAX_DRIVE_FRAMES = 2000
TRAFFIC_VEHICLES = 0
