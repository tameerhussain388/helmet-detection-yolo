"""
convert_voc_to_yolo.py
Converts Pascal VOC XML annotations to normalized YOLO format (.txt).

Pascal VOC format (absolute pixel coordinates):
  [xmin, ymin, xmax, ymax] on an image of size (W, H)

YOLO format (normalized coordinates between 0.0 and 1.0):
  <class_id> <x_center> <y_center> <width> <height>
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Class mapping: map your dataset labels to contiguous zero-indexed integers
CLASS_MAPPING = {
    "helmet": 0,
    "hard_hat": 0,
    "head": 1,
    "no_helmet": 1,
    "person": 2
}

def convert_voc_box_to_yolo(size, box):
    """
    Converts absolute VOC bbox [xmin, ymin, xmax, ymax]
    to normalized YOLO bbox [x_center, y_center, width, height].
    
    Args:
        size: tuple of (img_width, img_height)
        box: tuple of (xmin, ymin, xmax, ymax)
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # Calculate box center and spatial extent
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    # Normalize by image dimensions
    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh
    
    # Clamp to [0.0, 1.0] to prevent slight rounding out-of-bounds
    return (max(0.0, min(1.0, x)),
            max(0.0, min(1.0, y)),
            max(0.0, min(1.0, w)),
            max(0.0, min(1.0, h)))

def convert_voc_file(xml_path, output_dir):
    """Parses a single VOC XML file and writes corresponding YOLO .txt file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size_elem = root.find('size')
    img_w = int(size_elem.find('width').text)
    img_h = int(size_elem.find('height').text)
    
    if img_w <= 0 or img_h <= 0:
        return
        
    out_txt_name = Path(xml_path).stem + ".txt"
    out_txt_path = os.path.join(output_dir, out_txt_name)
    
    yolo_lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text.lower().strip()
        if cls_name not in CLASS_MAPPING:
            continue
        cls_id = CLASS_MAPPING[cls_name]
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Ensure valid coordinates
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        
        x_c, y_c, w, h = convert_voc_box_to_yolo((img_w, img_h), (xmin, ymin, xmax, ymax))
        yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        
    with open(out_txt_path, 'w') as f:
        f.write("\n".join(yolo_lines))

if __name__ == "__main__":
    print("Pascal VOC to YOLO conversion script ready.")
