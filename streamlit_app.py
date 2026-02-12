"""
Streamlit GUI for Real-Time Object Detection
============================================
A web-based interface for the object detection system

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import time
from utils.detector import ObjectDetector
from utils.fps_calculator import FPSCalculator


# Page configuration
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitObjectDetection:
    """Streamlit-based object detection application"""
    
    def __init__(self):
        self.detector = None
        self.fps_calc = FPSCalculator()
    
    def load_model(self, config_path, weights_path, names_path, input_size):
        """Load YOLO model"""
        try:
            self.detector = ObjectDetector(
                config_path=config_path,
                weights_path=weights_path,
                names_path=names_path,
                input_size=input_size
            )
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def detect_and_annotate(self, frame, confidence_threshold, nms_threshold):
        """Detect objects and annotate frame"""
        if self.detector is None:
            return frame, []
        
        # Detect objects
        detections = self.detector.detect(
            frame, 
            confidence_threshold, 
            nms_threshold
        )
        
        # Draw detections
        for detection in detections:
            x, y, w, h = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            color = detection['color']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(frame, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            cv2.putText(frame, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, detections


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🎯 Real-Time Object Detection</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'app' not in st.session_state:
        st.session_state.app = StreamlitObjectDetection()
        st.session_state.model_loaded = False
        st.session_state.running = False
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("📦 Model Settings")
    
    config_path = st.sidebar.text_input(
        "Config Path",
        value="models/yolov4.cfg"
    )
    
    weights_path = st.sidebar.text_input(
        "Weights Path",
        value="models/yolov4.weights"
    )
    
    names_path = st.sidebar.text_input(
        "Names Path",
        value="models/coco.names"
    )
    
    input_size = st.sidebar.selectbox(
        "Input Size",
        options=[320, 416, 608],
        index=1
    )
    
    # Detection settings
    st.sidebar.subheader("🎚️ Detection Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    nms_threshold = st.sidebar.slider(
        "NMS Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05
    )
    
    camera_id = st.sidebar.number_input(
        "Camera ID",
        min_value=0,
        max_value=10,
        value=0
    )
    
    # Load model button
    if st.sidebar.button("🚀 Load Model"):
        with st.spinner("Loading model..."):
            success = st.session_state.app.load_model(
                config_path, weights_path, names_path, input_size
            )
            if success:
                st.session_state.model_loaded = True
                st.sidebar.success("✅ Model loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load model")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Statistics")
        fps_metric = st.empty()
        detections_metric = st.empty()
        status_metric = st.empty()
        
        st.subheader("🎮 Controls")
        start_button = st.button("▶️ Start Detection")
        stop_button = st.button("⏹️ Stop Detection")
        
        st.subheader("ℹ️ Information")
        st.info("""
        **How to use:**
        1. Configure model paths in sidebar
        2. Click 'Load Model'
        3. Adjust detection settings
        4. Click 'Start Detection'
        5. Press 'Stop Detection' to stop
        """)
    
    # Start detection
    if start_button and st.session_state.model_loaded:
        st.session_state.running = True
    
    if stop_button:
        st.session_state.running = False
    
    # Detection loop
    if st.session_state.running and st.session_state.model_loaded:
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            st.error(f"❌ Could not open camera {camera_id}")
            st.session_state.running = False
        else:
            status_metric.success("🟢 Detection Active")
            
            while st.session_state.running:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to grab frame")
                    break
                
                # Detect and annotate
                annotated_frame, detections = st.session_state.app.detect_and_annotate(
                    frame, confidence_threshold, nms_threshold
                )
                
                # Update FPS
                fps = st.session_state.app.fps_calc.update()
                
                # Display metrics
                fps_metric.metric("FPS", f"{fps:.1f}")
                detections_metric.metric("Objects Detected", len(detections))
                
                # Display frame
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Small delay to prevent UI freezing
                time.sleep(0.01)
            
            cap.release()
            status_metric.warning("🟡 Detection Stopped")
    
    elif not st.session_state.model_loaded:
        status_metric.warning("⚠️ Please load model first")
    else:
        status_metric.info("⏸️ Detection Inactive")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit and OpenCV</p>
            <p>🎯 Real-Time Object Detection System v1.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()