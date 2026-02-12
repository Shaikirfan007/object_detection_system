"""
Utils Package
=============
Utility modules for the real-time object detection system
"""

from .detector import ObjectDetector
from .fps_calculator import FPSCalculator
from .video_capture import VideoCapture

__all__ = ['ObjectDetector', 'FPSCalculator', 'VideoCapture']