"""
detect.py
Runs YOLOv8 object detection on any input image, directory, or video.

Usage:
  python detect.py --source sample.jpg
  python detect.py --source 0               # runs on your webcam in real-time!
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_detection(source, weights='yolov8n.pt', conf=0.25):
    print(f"Loading YOLOv8 model ({weights})...")
    model = YOLO(weights)
    
    print(f"Running detection on: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        project='results',
        name='predictions',
        exist_ok=True
    )
    
    print("\n" + "="*50)
    print("DETECTION RESULTS:")
    print("="*50)
    for r in results:
        boxes = r.boxes
        print(f"Found {len(boxes)} object(s):")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0]) * 100
            coords = [round(float(c), 1) for c in box.xyxy[0]]
            print(f" - {cls_name}: {confidence:.1f}% confidence | bbox: {coords}")
            
    print("="*50)
    print("Annotated image saved in 'results/predictions/'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection")
    parser.add_argument('--source', type=str, default='https://ultralytics.com/images/bus.jpg', help='Image path, URL, or 0 for webcam')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Model weights path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    run_detection(args.source, args.weights, args.conf)
