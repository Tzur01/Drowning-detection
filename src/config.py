"""
Configuration settings for the drowning detection system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = PROJECT_ROOT / "videos"
SOUNDS_DIR = PROJECT_ROOT / "sound"

# Model files
MODEL_PATH = MODELS_DIR / "model.pth"
LABEL_BINARIZER_PATH = MODELS_DIR / "lb.pkl"

# Detection settings
class DetectionConfig:
    """Configuration for detection parameters."""
    
    # Image processing
    IMAGE_SIZE = 224
    FRAME_LIMIT = 500  # Maximum frames to process
    
    # Distance threshold for multiple person detection
    MIN_DISTANCE_THRESHOLD = 50
    
    # Confidence thresholds
    DETECTION_CONFIDENCE = 0.5
    
    # Alert settings
    ENABLE_VOICE_ALERTS = True
    ENABLE_AUDIO_ALERTS = True
    AUDIO_FILE = SOUNDS_DIR / "alarm.mp3"
    
    # Display settings
    WINDOW_NAME = "Drowning Detection System"
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class ModelConfig:
    """Configuration for the CNN model."""
    
    # CNN Architecture
    CONV1_CHANNELS = 16
    CONV2_CHANNELS = 32
    CONV3_CHANNELS = 64
    CONV4_CHANNELS = 128
    FC1_SIZE = 256
    
    # Training parameters (for reference)
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
