"""
Video Capture Module
===================
Handles webcam initialization and frame capture with error handling
"""

import cv2
import sys


class VideoCapture:
    """Wrapper for cv2.VideoCapture with error handling"""
    
    def __init__(self, camera_id=0):
        """
        Initialize video capture
        
        Args:
            camera_id: Camera device ID (0 for default camera)
        """
        print(f"📷 Initializing camera (ID: {camera_id})...")
        
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            print("\nTroubleshooting:")
            print("1. Make sure your webcam is connected")
            print("2. Check if another application is using the camera")
            print("3. Try a different camera ID (e.g., --camera 1)")
            print("4. Check camera permissions")
            sys.exit(1)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ Camera initialized successfully!")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - FPS: {fps}")
    
    def read(self):
        """
        Read a frame from the camera
        
        Returns:
            Tuple of (success, frame)
        """
        return self.cap.read()
    
    def release(self):
        """Release the camera"""
        if self.cap is not None:
            self.cap.release()
            print("📷 Camera released")
    
    def __del__(self):
        """Destructor to ensure camera is released"""
        self.release()