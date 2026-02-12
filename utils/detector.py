"""
Object Detector Module
=====================
Handles YOLO model loading and object detection logic
"""

import cv2
import numpy as np


class ObjectDetector:
    """Wrapper class for YOLO object detection"""
    
    def __init__(self, config_path, weights_path, names_path, input_size=416):
        """
        Initialize the object detector
        
        Args:
            config_path: Path to YOLO configuration file
            weights_path: Path to YOLO weights file
            names_path: Path to class names file
            input_size: Input size for the network
        """
        print("📦 Loading YOLO model...")
        
        # Load class names
        with open(names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))
        
        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Set backend and target
        # For CPU: cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
        # For GPU: cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.input_size = input_size
        
        print(f"✅ Model loaded successfully!")
        print(f"   - Classes: {len(self.class_names)}")
        print(f"   - Input size: {input_size}x{input_size}")
    
    def detect(self, frame, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
        
        Returns:
            List of detection dictionaries containing:
                - box: (x, y, width, height)
                - label: class name
                - confidence: detection confidence
                - color: BGR color tuple
        """
        height, width = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            (self.input_size, self.input_size), 
            swapRB=True, 
            crop=False
        )
        
        # Set input and perform forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            confidence_threshold, 
            nms_threshold
        )
        
        # Prepare final detections
        detections = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = self.class_names[class_ids[i]]
                confidence = confidences[i]
                color = tuple(map(int, self.colors[class_ids[i]]))
                
                detections.append({
                    'box': (x, y, w, h),
                    'label': label,
                    'confidence': confidence,
                    'color': color,
                    'class_id': class_ids[i]
                })
        
        return detections