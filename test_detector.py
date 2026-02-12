"""
Unit tests for the object detector module
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from utils.detector import ObjectDetector


class TestObjectDetector:
    """Test suite for ObjectDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for tests"""
        return ObjectDetector(
            config_path='models/yolov4.cfg',
            weights_path='models/yolov4.weights',
            names_path='models/coco.names',
            input_size=416
        )
    
    def test_detector_initialization(self, detector):
        """Test that detector initializes correctly"""
        assert detector is not None
        assert len(detector.class_names) == 80
        assert detector.input_size == 416
        assert detector.colors.shape[0] == 80
    
    def test_class_names_loaded(self, detector):
        """Test that COCO class names are loaded"""
        assert 'person' in detector.class_names
        assert 'car' in detector.class_names
        assert 'dog' in detector.class_names
        assert len(detector.class_names) == 80
    
    def test_detection_output_format(self, detector):
        """Test that detections have correct format"""
        # Create dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(dummy_frame)
        
        # Should return list
        assert isinstance(detections, list)
        
        # Each detection should have required fields
        for det in detections:
            assert 'box' in det
            assert 'label' in det
            assert 'confidence' in det
            assert 'color' in det
            
            # Verify box format (x, y, w, h)
            assert len(det['box']) == 4
            assert all(isinstance(v, int) for v in det['box'])
            
            # Verify confidence is float between 0 and 1
            assert isinstance(det['confidence'], float)
            assert 0 <= det['confidence'] <= 1
            
            # Verify label is string
            assert isinstance(det['label'], str)
            assert det['label'] in detector.class_names
    
    def test_confidence_threshold(self, detector):
        """Test that confidence threshold filters detections"""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        low_conf = detector.detect(dummy_frame, confidence_threshold=0.9)
        high_conf = detector.detect(dummy_frame, confidence_threshold=0.1)
        
        # Lower threshold should detect same or more
        assert len(high_conf) >= len(low_conf)
    
    def test_different_input_sizes(self):
        """Test detector with different input sizes"""
        for size in [320, 416, 608]:
            detector = ObjectDetector(
                config_path='models/yolov4.cfg',
                weights_path='models/yolov4.weights',
                names_path='models/coco.names',
                input_size=size
            )
            assert detector.input_size == size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])