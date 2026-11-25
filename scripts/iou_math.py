"""
iou_math.py
First-principles implementation of Intersection over Union (IoU).

Mathematical Definition:
  IoU = Area of Overlap / Area of Union
      = Area(A ∩ B) / (Area(A) + Area(B) - Area(A ∩ B))

Box format: [xmin, ymin, xmax, ymax]
"""

def compute_iou(box_a, box_b):
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box_a: [xmin, ymin, xmax, ymax]
        box_b: [xmin, ymin, xmax, ymax]
    Returns:
        float: IoU value in range [0.0, 1.0]
    """
    # 1. Determine intersection coordinates
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    # If there is no overlap, intersection is 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 2. Area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 3. Area of both bounding boxes
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # 4. Area of Union = Area(A) + Area(B) - Area(A ∩ B)
    union_area = float(box_a_area + box_b_area - intersection_area)

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

# Verification test with a concrete hand-calculated example:
# Box A: [50, 50, 150, 150] -> Width = 100, Height = 100, Area = 10,000
# Box B: [100, 100, 200, 200] -> Width = 100, Height = 100, Area = 10,000
# Intersection: x in [100, 150] (w=50), y in [100, 150] (h=50) -> Area = 2,500
# Union: 10,000 + 10,000 - 2,500 = 17,500
# Expected IoU: 2,500 / 17,500 = 1 / 7 ≈ 0.142857

if __name__ == "__main__":
    box1 = [50, 50, 150, 150]
    box2 = [100, 100, 200, 200]
    result = compute_iou(box1, box2)
    print(f"Hand calculation: 2500 / 17500 = {1/7:.6f}")
    print(f"compute_iou:     {result:.6f}")
    assert abs(result - (1/7)) < 1e-5, "IoU math test failed!"
    print("SUCCESS: IoU calculation mathematically verified from first principles!")
