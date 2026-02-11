import os
from pathlib import Path

# Project Root
# Project Root
# robustly find root regardless of where script is run
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Data Specifics
IMG_SIZE = 512  # Original size, might resize
NUM_CLASSES = 3
CLASS_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}

# Random Seed
SEED = 42
