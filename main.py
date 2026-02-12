"""
Real-Time Object Detection System
==================================
A complete computer vision project for detecting objects in real-time using webcam.
Uses YOLOv4 with OpenCV DNN module for fast and accurate detection.

Author: CV Intern Project
Date: 2026
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from utils.detector import ObjectDetector
from utils.fps_calculator import FPSCalculator
from utils.video_capture import VideoCapture


class RealTimeObjectDetection:
    """Main application class for real-time object detection"""
    
    def __init__(self, config_path, weights_path, names_path,
                 confidence_threshold=0.5, nms_threshold=0.4,
                 input_size=416, camera_id=0, save_output=False):
        """
        Initialize the object detection system

        Args:
            config_path: Path to model configuration file
            weights_path: Path to model weights file
            names_path: Path to class names file
            confidence_threshold: Minimum confidence for detection (0-1)
            nms_threshold: Non-maximum suppression threshold
            input_size: Input size for the network (416, 608, etc.)
            camera_id: Camera device ID (0=default webcam)
            save_output: Whether to save detected frames
        """
        print("🚀 Initializing Real-Time Object Detection System...")

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.save_output = save_output

        self.detector = ObjectDetector(
            config_path=config_path,
            weights_path=weights_path,
            names_path=names_path,
            input_size=input_size
        )

        self.fps_calc = FPSCalculator()

        self.video_capture = VideoCapture(camera_id=camera_id)

        if self.save_output:
            self.output_dir = Path("outputs/detected_frames")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.frame_count = 0

        print("✅ System initialized successfully!")

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame

        Args:
            frame: Input frame
            detections: List of detection dictionaries

        Returns:
            Annotated frame
        """
        for detection in detections:
            x, y, w, h = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            color = detection['color']

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label_text = f"{label}: {confidence:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width, y),
                color,
                -1
            )

            cv2.putText(
                frame,
                label_text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame

    def draw_info_panel(self, frame, fps, num_detections):
        """
        Draw information panel on frame

        Args:
            frame: Input frame
            fps: Current FPS
            num_detections: Number of objects detected

        Returns:
            Frame with info panel
        """
        height, width = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        det_text = f"Objects Detected: {num_detections}"
        cv2.putText(frame, det_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, "Press 'Q' to quit | 'S' to save",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def save_frame(self, frame):
        """Save current frame to output directory"""
        if self.save_output:
            filename = self.output_dir / f"frame_{self.frame_count:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            self.frame_count += 1
            print(f"💾 Saved: {filename}")

    def run(self):
        """Main detection loop"""
        print("\n📹 Starting real-time detection...")
        print("Press 'Q' to quit")
        print("Press 'S' to save current frame")
        print("-" * 50)

        try:
            while True:
                ret, frame = self.video_capture.read()

                if not ret:
                    print("❌ Failed to grab frame")
                    break

                detections = self.detector.detect(
                    frame,
                    self.confidence_threshold,
                    self.nms_threshold
                )

                annotated_frame = self.draw_detections(frame.copy(), detections)

                fps = self.fps_calc.update()

                annotated_frame = self.draw_info_panel(
                    annotated_frame,
                    fps,
                    len(detections)
                )

                cv2.imshow('Real-Time Object Detection', annotated_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\n🛑 Quitting...")
                    break
                elif key == ord('s') or key == ord('S'):
                    self.save_frame(annotated_frame)

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")

        finally:
            self.video_capture.release()
            cv2.destroyAllWindows()
            print("✅ Cleanup complete")

            if self.save_output and self.frame_count > 0:
                print(f"📁 Saved {self.frame_count} frames to {self.output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Real-Time Object Detection System'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='models/yolov4.cfg',
        help='Path to model configuration file'
    )

    parser.add_argument(
        '--weights',
        type=str,
        default='models/yolov4.weights',
        help='Path to model weights file'
    )

    parser.add_argument(
        '--names',
        type=str,
        default='models/coco.names',
        help='Path to class names file'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold (0-1)'
    )

    parser.add_argument(
        '--nms',
        type=float,
        default=0.4,
        help='Non-maximum suppression threshold (0-1)'
    )

    parser.add_argument(
        '--input-size',
        type=int,
        default=416,
        choices=[320, 416, 608],
        help='Input size for the network'
    )

    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (0=default, 1=external USB, etc.)'
    )

    parser.add_argument(
        '--save-output',
        action='store_true',
        help='Save detected frames to output directory'
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    weights_path = Path(args.weights)
    names_path = Path(args.names)

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Please download YOLOv4 configuration file.")
        return

    if not weights_path.exists():
        print(f"❌ Weights file not found: {weights_path}")
        print("Please download YOLOv4 weights file.")
        return

    if not names_path.exists():
        print(f"❌ Names file not found: {names_path}")
        print("Please download COCO class names file.")
        return

    app = RealTimeObjectDetection(
        config_path=str(config_path),
        weights_path=str(weights_path),
        names_path=str(names_path),
        confidence_threshold=args.confidence,
        nms_threshold=args.nms,
        input_size=args.input_size,
        camera_id=args.camera,
        save_output=args.save_output
    )

    app.run()


if __name__ == "__main__":
    main()