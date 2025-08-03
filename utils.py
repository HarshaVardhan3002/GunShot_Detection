"""
Utility functions for gunshot localization system.
"""
import logging
import time
from typing import Any, Dict
import numpy as np


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("gunshot_localizer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_distance_2d(point1: tuple, point2: tuple) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_distance_3d(point1: tuple, point2: tuple) -> float:
    """Calculate 3D Euclidean distance between two points."""
    return np.sqrt(
        (point1[0] - point2[0])**2 + 
        (point1[1] - point2[1])**2 + 
        (point1[2] - point2[2])**2
    )


def format_coordinates(x: float, y: float, z: float = 0.0) -> str:
    """Format coordinates for display."""
    if z == 0.0:
        return f"(X: {x:.1f} m, Y: {y:.1f} m)"
    else:
        return f"(X: {x:.1f} m, Y: {y:.1f} m, Z: {z:.1f} m)"


def get_timestamp() -> float:
    """Get current timestamp."""
    return time.time()


def validate_audio_data(audio_data: np.ndarray, expected_channels: int = 8) -> bool:
    """
    Validate audio data format.
    
    Args:
        audio_data: Audio data array
        expected_channels: Expected number of channels
        
    Returns:
        True if valid, False otherwise
    """
    if audio_data is None:
        return False
    
    if len(audio_data.shape) != 2:
        return False
    
    if audio_data.shape[1] != expected_channels:
        return False
    
    return True