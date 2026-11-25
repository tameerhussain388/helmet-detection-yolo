# Theoretical Foundations & Interview Defense Notes

Mathematical derivations and conceptual explanations for the Safety-Helmet Object Detection project.

---

## 1. Intersection over Union (IoU)
$$\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$

### Numerical Example (To solve on the blackboard during an interview):
- **Box A:** $[50, 50, 150, 150] \implies \text{Width} = 100, \text{Height} = 100, \text{Area} = 10,000$
- **Box B:** $[100, 100, 200, 200] \implies \text{Width} = 100, \text{Height} = 100, \text{Area} = 10,000$
- **Intersection:** $x \in [100, 150]$ (width = 50), $y \in [100, 150]$ (height = 50) $\implies \text{Area} = 2,500$
- **Union:** $10,000 + 10,000 - 2,500 = 17,500$
- **IoU:** $\frac{2,500}{17,500} = \frac{1}{7} \approx 0.142857$

---

## 2. Non-Maximum Suppression (NMS)
Detectors predict dozens of candidate bounding boxes per object. NMS prunes redundant boxes:
1. Sort all predicted boxes by confidence score in descending order.
2. Select the box with the highest confidence ($B_{\text{max}}$) and append it to final detections.
3. Remove all remaining boxes $B_i$ where $\text{IoU}(B_{\text{max}}, B_i) > \tau_{\text{nms}}$ (typically 0.45 to 0.65).
4. Repeat for all remaining candidate boxes.

---

## 3. mAP@0.5 vs. mAP@0.5:0.95
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (What proportion of detected helmets were actual helmets?)
- **Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (What proportion of actual helmets did the model find?)
- **AP (Average Precision):** Area under the Precision-Recall curve: $\int_0^1 P(R) dR$.
- **mAP@0.5:** Mean AP across all classes at IoU threshold = 0.50 (forgiving of slight boundary misalignment).
- **mAP@0.5:0.95:** Mean AP averaged over 10 IoU thresholds ($0.50, 0.55, \dots, 0.95$). This metric heavily penalizes loose or inaccurate bounding box boundaries and is the official benchmark of COCO.

---

## 4. YOLOv8 Loss Formulations
YOLOv8 optimizes three distinct objective functions:

### 1. Complete IoU Box Loss ($\mathcal{L}_{\text{CIoU}}$)
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
- $\rho(b, b^{\text{gt}})$: Euclidean distance between bounding box centers.
- $c$: Diagonal length of the smallest enclosing box covering both boxes.
- $\alpha v$: Aspect ratio consistency penalty term.

### 2. Distribution Focal Loss ($\mathcal{L}_{\text{DFL}}$)
Anchor-free models predict continuous distances to the 4 edges. DFL models continuous coordinates as a discrete probability distribution over neighboring integer grid points, allowing the network to express uncertainty around blurry, motion-blurred, or occluded boundaries.

### 3. Classification Loss ($\mathcal{L}_{\text{cls}}$)
Binary Cross-Entropy (BCE) evaluated across all classes for object classification.

---

## 5. Interview Defense Q&A

### Q: Why is detecting small objects harder than large objects?
Small objects occupy very few pixels ($< 32 \times 32$). As the image passes through successive strided convolutions and pooling layers, spatial resolution decreases by $32\times$. An 18-pixel helmet is reduced to less than one feature cell at deep layers. Even with PAN-FPN feature pyramids merging shallow layers, fine spatial details (such as helmet contours and chinstraps) suffer severe loss.

### Q: How is multi-object tracking different from object detection?
Object detection operates on individual frames without temporal awareness ($X_t \to \text{Boxes}_t$). Multi-object tracking (e.g. ByteTrack) maintains object identities over time ($X_t \to \text{Tracked ID}_k$). It uses Kalman filters to model object motion vectors and Hungarian matching to pair frame-by-frame detections, preventing ID switches when cameras move or workers cross paths.
