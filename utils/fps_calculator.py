"""
FPS Calculator Module
====================
Calculates frames per second for performance monitoring
"""

import time


class FPSCalculator:
    """Calculate and smooth FPS measurements"""
    
    def __init__(self, smoothing_factor=0.9):
        """
        Initialize FPS calculator
        
        Args:
            smoothing_factor: Exponential moving average factor (0-1)
                             Higher = smoother but less responsive
        """
        self.smoothing_factor = smoothing_factor
        self.fps = 0
        self.prev_time = time.time()
    
    def update(self):
        """
        Update FPS calculation
        
        Returns:
            Current FPS value (smoothed)
        """
        current_time = time.time()
        elapsed = current_time - self.prev_time
        
        # Calculate instantaneous FPS
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
        else:
            instant_fps = 0
        
        # Apply exponential moving average for smoothing
        if self.fps == 0:
            self.fps = instant_fps
        else:
            self.fps = (self.smoothing_factor * self.fps + 
                       (1 - self.smoothing_factor) * instant_fps)
        
        self.prev_time = current_time
        
        return self.fps
    
    def reset(self):
        """Reset FPS calculator"""
        self.fps = 0
        self.prev_time = time.time()