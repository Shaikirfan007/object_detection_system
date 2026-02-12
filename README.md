# 🎯 Real-Time Object Detection System

A professional-grade real-time object detection application using YOLOv4 and OpenCV. Perfect for Computer Vision internship portfolios and final-year projects.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- ✅ **Real-time object detection** from webcam feed
- ✅ **80 object classes** (COCO dataset)
- ✅ **Bounding boxes** with confidence scores
- ✅ **Live FPS counter** for performance monitoring
- ✅ **Configurable confidence thresholds**
- ✅ **Frame saving** capability
- ✅ **Modular, clean code** structure
- ✅ **Comprehensive error handling**
- ✅ **Command-line interface** with arguments

## 📁 Project Structure

```
realtime_object_detection/
│
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── models/                          # Model files directory
│   ├── yolov4.cfg                  # YOLOv4 configuration (download)
│   ├── yolov4.weights              # YOLOv4 weights (download)
│   └── coco.names                  # COCO class names (download)
│
├── utils/                           # Utility modules
│   ├── __init__.py                 # Package initializer
│   ├── detector.py                 # Object detection logic
│   ├── fps_calculator.py           # FPS calculation
│   └── video_capture.py            # Camera handling
│
└── outputs/                         # Output directory
    └── detected_frames/            # Saved frames (created at runtime)
```

## 🚀 Quick Start

### 1️⃣ Prerequisites

- Python 3.8 or higher
- Webcam/camera device
- ~250 MB free disk space (for model weights)

### 2️⃣ Installation

**Clone or download this project:**
```bash
cd realtime_object_detection
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Model Files

You need to download three files and place them in the `models/` directory:

#### **Method 1: Download using wget/curl (Linux/Mac)**

```bash
# Create models directory
mkdir -p models

# Download YOLOv4 configuration
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg -O models/yolov4.cfg

# Download YOLOv4 weights (~250 MB)
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights -O models/yolov4.weights

# Download COCO class names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O models/coco.names
```

#### **Method 2: Manual Download (Windows/All)**

1. **yolov4.cfg** (Configuration file)
   - URL: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
   - Save to: `models/yolov4.cfg`

2. **yolov4.weights** (Pre-trained weights ~250 MB)
   - URL: https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights
   - Save to: `models/yolov4.weights`

3. **coco.names** (Class names)
   - URL: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
   - Save to: `models/coco.names`

#### **Method 3: Python Download Script**

Create and run this script:

```python
# download_models.py
import urllib.request
from pathlib import Path

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

files = {
    "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
    "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

for filename, url in files.items():
    filepath = models_dir / filename
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"✅ Downloaded: {filepath}")

print("\n🎉 All model files downloaded successfully!")
```

Run: `python download_models.py`

### 4️⃣ Run the Application

**Basic usage:**
```bash
python main.py
```

**With custom parameters:**
```bash
python main.py --confidence 0.6 --nms 0.4 --input-size 416
```

**Save detected frames:**
```bash
python main.py --save-output
```

## ⚙️ Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `models/yolov4.cfg` | Path to model configuration file |
| `--weights` | str | `models/yolov4.weights` | Path to model weights file |
| `--names` | str | `models/coco.names` | Path to class names file |
| `--confidence` | float | `0.5` | Minimum confidence threshold (0-1) |
| `--nms` | float | `0.4` | Non-maximum suppression threshold (0-1) |
| `--input-size` | int | `416` | Network input size (320/416/608) |
| `--save-output` | flag | `False` | Save detected frames to output directory |

## 🎮 Keyboard Controls

| Key | Action |
|-----|--------|
| **Q** | Quit application |
| **S** | Save current frame |

## 📊 Performance Tips

### For Higher FPS:
- Use smaller input size: `--input-size 320`
- Increase confidence threshold: `--confidence 0.7`
- Close other applications

### For Better Accuracy:
- Use larger input size: `--input-size 608`
- Lower confidence threshold: `--confidence 0.3`
- Ensure good lighting

### GPU Acceleration (Optional):
If you have a CUDA-compatible GPU, modify `utils/detector.py`:

```python
# Change these lines in ObjectDetector.__init__()
self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

Then install CUDA-enabled OpenCV:
```bash
pip install opencv-contrib-python
```

## 🔍 Detectable Objects (80 Classes)

The system can detect 80 different object classes from the COCO dataset:

**People & Animals:**
person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Objects:**
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Food:**
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Furniture & Electronics:**
chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## 🛠️ Troubleshooting

### Camera not opening?
```python
# Try different camera IDs in main.py
# For external webcams, try camera_id=1 or 2
self.video_capture = VideoCapture(camera_id=1)
```

### Low FPS?
- Reduce input size to 320
- Close other applications
- Check CPU usage
- Consider GPU acceleration

### Model files not found?
- Verify files are in `models/` directory
- Check file names match exactly
- Re-download if corrupted

### Import errors?
```bash
pip install --upgrade opencv-python numpy
```

## 📚 Code Explanation

### Main Components:

1. **main.py** - Application orchestration
   - Initializes all components
   - Manages the main detection loop
   - Handles user input and display

2. **utils/detector.py** - Detection engine
   - Loads YOLO model
   - Performs object detection
   - Applies NMS (Non-Maximum Suppression)

3. **utils/fps_calculator.py** - Performance monitoring
   - Calculates frames per second
   - Applies smoothing for stable readings

4. **utils/video_capture.py** - Camera interface
   - Initializes webcam
   - Handles errors gracefully
   - Manages frame capture

## 🎓 Learning Outcomes

By studying and running this project, you will learn:

- ✅ Real-time computer vision applications
- ✅ Deep learning model integration
- ✅ OpenCV DNN module usage
- ✅ Object detection algorithms (YOLO)
- ✅ Python project structuring
- ✅ Command-line interface design
- ✅ Error handling and debugging
- ✅ Performance optimization techniques

## 🚀 Future Enhancements (Ideas for Your Portfolio)

1. **Object Tracking**: Add centroid tracking or DeepSORT
2. **Multi-object Tracking**: Track multiple objects across frames
3. **GUI Interface**: Build with Tkinter or Streamlit
4. **Video File Input**: Support video file processing
5. **ROI Selection**: Allow region of interest selection
6. **Alert System**: Trigger alerts for specific objects
7. **Cloud Integration**: Upload detections to cloud storage
8. **Mobile App**: Build companion mobile app
9. **Multiple Models**: Support YOLOv5, YOLOv8, SSD, etc.
10. **Custom Training**: Train on custom datasets

## 📝 Common Interview Questions

**Q: Why YOLO over other models?**
A: YOLO provides excellent balance between speed and accuracy, making it ideal for real-time applications. It processes the entire image in one pass, unlike R-CNN family which requires multiple passes.

**Q: What is NMS and why is it needed?**
A: Non-Maximum Suppression removes redundant overlapping bounding boxes, keeping only the most confident detection for each object.

**Q: How does confidence threshold affect results?**
A: Higher threshold (e.g., 0.7) = fewer false positives but might miss some objects. Lower threshold (e.g., 0.3) = more detections but more false positives.

**Q: Can this run on edge devices?**
A: Yes, but consider lighter models like YOLOv4-tiny or MobileNet-SSD for better performance on resource-constrained devices.

## 📄 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📧 Contact

For questions or feedback, create an issue in the repository.

---

**⭐ If you found this project helpful, please star it! ⭐**

Good luck with your internship applications! 🚀