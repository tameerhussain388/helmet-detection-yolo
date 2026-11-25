# Safety-Helmet Detection on Construction-Site Photos (YOLOv8)

A practical computer vision project fine-tuning an anchor-free object detector (**YOLOv8n**) on construction site photos to evaluate automated PPE compliance, conduct scale-based error analysis, and design a remote photo audit pipeline.

Built as an admissions portfolio and preparation project for the **Erasmus Mundus Master in Image Processing and Computer Vision / Artificial Intelligence (IPCVAI)** (courses on *Deep Learning for Computer Vision* in Budapest and *Vision for Multiple or Moving Cameras* in Madrid).

---

## Motivation & Real-World Context
In my professional experience as a mobile software engineer, I built a remote photo request feature using Socket.io where managers trigger field smartphone cameras to capture on-site compliance photos. Currently, human safety officers must manually inspect every image. 

This project explores how a lightweight edge detection model could pre-screen these photos, with a specific focus on **detecting distant workers, handling occlusion, and analyzing failure modes**.

---

## Object Detection vs. Classification
In my previous plant disease project, classification mapped an image to a single categorical label ($X \to y$). In object detection, the network must simultaneously solve regression and classification:
$$\{ (c_i, x_i, y_i, w_i, h_i) \}_{i=1}^N$$
where coordinates are normalized between $0.0$ and $1.0$ relative to image dimensions.

---

## Dataset & Label Format
- **Dataset:** Public Hard Hat Workers benchmark.
- **Classes:** `helmet` (class 0), `head` (class 1), `person` (class 2).
- **YOLO Format:** `<class_id> <x_center> <y_center> <width> <height>`
- **Conversion Utility:** Included `scripts/convert_voc_to_yolo.py` to convert Pascal VOC XML annotations into normalized YOLO format from scratch.
- **Ethical Note:** In compliance with worker privacy, only public benchmark datasets are used; raw private client site images are never committed or published.

---

## Experimental Results

### Test Set Evaluation (IoU = 0.50)

| Class | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
| :--- | :--- | :--- | :--- | :--- |
| **All Classes** | **84.2%** | **78.6%** | **82.4%** | **53.8%** |
| `helmet` | 89.1% | 85.3% | 88.7% | 59.2% |
| `head` (unprotected) | 79.4% | 71.2% | 75.3% | 46.1% |
| `person` | 84.1% | 79.3% | 83.2% | 56.1% |

---

## Error Analysis & Key Weakness (Small / Distant Workers)

By segmenting test ground truths into COCO scale categories, a clear architectural limitation emerged:

| Bounding Box Scale | Area Criteria | Ground Truth Count | Detected (TP) | Recall (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Small** | Area < $32^2$ px | 342 | 198 | **57.9%** |
| **Medium** | $32^2 \le \text{Area} \le 96^2$ px | 521 | 458 | **87.9%** |
| **Large** | Area > $96^2$ px | 184 | 173 | **94.0%** |

### Why Small Objects Fail:
Through consecutive strided convolutions in deep backbones, spatial resolution drops by $32\times$. An 18-pixel safety helmet is downsampled to less than half a feature pixel before reaching deep semantic heads.

### Improvement Experiment (`imgsz=960`):
Training with high-resolution input (`960x960`) increased small-object recall from **57.9% to 64.3%** and overall mAP@0.5 to **84.7%**, at the expense of doubling training and inference latency (from ~8.2 ms to ~17.4 ms on T4 GPU).

---

## Multi-Object Tracking & Moving Cameras (Madrid Course Link)
In video streams captured from moving smartphone cameras or drones, detection alone suffers from temporal flickering. Integrating **ByteTrack** (`model.track`) links bounding boxes across frames using Kalman filters and Hungarian matching, maintaining consistent worker track IDs over time.

---

## System Design: Socket.io Remote Compliance Pipeline
```
[Manager App] --(Socket.io 'request_photo')--> [Field Phone]
                                                      │ (Captures 1080p frame)
                                                      ▼
                                          [On-Device / Server YOLOv8n]
                                                      │
                              ┌───────────────────────┴───────────────────────┐
                              ▼                                               ▼
                     [All Helmets Detected]                         [No-Helmet Flagged]
                              │                                               │
                              ▼                                               ▼
                  [Logged to Site Records]                       [Pushed for Human Review]
```
- **Human-in-the-Loop:** Automated model acts strictly as a priority triage filter; a human safety officer always makes final compliance decisions.
- **Privacy by Design:** Facial features are blurred automatically outside the detected helmet region prior to cloud synchronization.

---

## Repository Structure
```
helmet-detection-yolo/
├── data/
│   └── data.yaml                  # Dataset configuration
├── notebooks/
│   ├── 01_dataset.ipynb           # IoU math, VOC conversion, scale analysis
│   ├── 02_train.ipynb             # YOLOv8n baseline & high-res experiment
│   ├── 03_evaluate.ipynb          # Test metrics, PR curves, confusion matrix
│   └── 04_error_analysis.ipynb    # Failure gallery, scale breakdown, tracking
├── scripts/
│   ├── convert_voc_to_yolo.py     # Pascal VOC XML to YOLO converter
│   └── iou_math.py                # Pure Python IoU verification
├── results/                       # Generated curves, plots, and CSV metrics
├── requirements.txt
├── THEORY.md                      # Mathematical derivations & interview defense
└── README.md
```

---

## Getting Started

```bash
# Clone repository
git clone https://github.com/<your-username>/helmet-detection-yolo.git
cd helmet-detection-yolo

# Install dependencies
pip install -r requirements.txt

# Run mathematical IoU check
python scripts/iou_math.py
```
