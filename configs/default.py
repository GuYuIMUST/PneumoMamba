"""
Default configuration for PneumonMamba training.
"""

# Model
MODEL_SIZE = "tiny"          # Options: "tiny", "small", "base"
NUM_CLASSES = 4              # Number of classification classes
IMG_SIZE = 224               # Input image size

# Training
BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 0.0001
SEED = 42

# Data
DATA_DIR = "/path/to/dataset"  # Root directory containing train/val/test splits
NUM_WORKERS = 8

# Checkpoints
SAVE_DIR = "../checkpoints"
SAVE_PATH = "../checkpoints/best_model.pth"

# Model variants
MODEL_CONFIGS = {
    "tiny": {
        "depths": [2, 2, 4, 2],
        "dims": [96, 192, 384, 768],
    },
    "small": {
        "depths": [2, 2, 8, 2],
        "dims": [96, 192, 384, 768],
    },
    "base": {
        "depths": [2, 2, 12, 2],
        "dims": [128, 256, 512, 1024],
    },
}
