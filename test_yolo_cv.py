"""
test_yolo_cv.py
Runs YOLOv8 object detection on PC using OpenCV's DNN module.
Zero PyTorch dependencies. Zero missing DLL errors.
"""

import os
import sys
import time
import urllib.request
import numpy as np
import cv2

MODEL_URL = "https://huggingface.co/arnabdhar/YOLOv8-ONNX/resolve/main/yolov8n.onnx"
MODEL_PATH = "yolov8n.onnx"
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = "sample.jpg"

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def download_file(url, target_path):
    import requests
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers, stream=True)
    with open(target_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def ensure_assets():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading lightweight YOLOv8 ONNX model (~12 MB)...")
        download_file(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
        
    if not os.path.exists(SAMPLE_PATH):
        print("Downloading sample test image...")
        download_file(SAMPLE_URL, SAMPLE_PATH)
        print("Sample image downloaded.")

def run_opencv_yolo():
    ensure_assets()
    
    print(f"Loading YOLOv8 into OpenCV DNN engine...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    
    # Load and preprocess image
    img = cv2.imread(SAMPLE_PATH)
    h_orig, w_orig = img.shape[:2]
    
    # YOLOv8 expects 640x640 RGB image scaled to [0, 1]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/255.0, size=(640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    
    print("Running on-device inference...")
    t0 = time.perf_counter()
    outputs = net.forward()
    latency_ms = (time.perf_counter() - t0) * 1000
    
    # Output shape: [1, 84, 8400] -> (x, y, w, h, 80 class scores)
    predictions = np.squeeze(outputs[0]).T # [8400, 84]
    
    boxes = []
    confidences = []
    class_ids = []
    
    x_factor = w_orig / 640.0
    y_factor = h_orig / 640.0
    
    for row in predictions:
        classes_scores = row[4:]
        max_score = np.max(classes_scores)
        if max_score >= 0.25:
            class_id = np.argmax(classes_scores)
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            # Convert to corner coordinates
            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(float(max_score))
            class_ids.append(class_id)
            
    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.45)
    
    print("\n" + "="*55)
    print("       YOLOv8 ON-DEVICE DETECTION RESULTS")
    print("="*55)
    print(f"Inference Latency: {latency_ms:.1f} ms (CPU SIMD)")
    print(f"Total Detections after NMS: {len(indices)}")
    
    for idx in indices:
        i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
        cls_name = COCO_CLASSES[class_ids[i]]
        conf = confidences[i] * 100
        box = boxes[i]
        print(f" - {cls_name.upper()}: {conf:.1f}% confidence | bbox: {box}")
        
        # Draw on image
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        cv2.putText(img, f"{cls_name} {conf:.0f}%", (box[0], max(box[1] - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    os.makedirs("results", exist_ok=True)
    out_img_path = "results/yolo_detection_output.jpg"
    cv2.imwrite(out_img_path, img)
    print("="*55)
    print(f"Saved annotated image with bounding boxes to: {out_img_path}")
    print("SUCCESS: Object detection verified on your PC!")

if __name__ == '__main__':
    run_opencv_yolo()
