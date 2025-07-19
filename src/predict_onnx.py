#!/usr/bin/env python3
import os
import cv2
import numpy as np
import onnxruntime as rt

# Configuration  
_MODEL_PATH  = "runs/train/yolo_mood/weights/best.onnx"
_SESSION     = rt.InferenceSession(_MODEL_PATH)
_INPUT_NAME  = _SESSION.get_inputs()[0].name

# Confirmed from mood_data.yaml
_CLASS_NAMES = ["surprised", "fear", "disgust", "smug", "sad", "angry", "neutral"]
_CONF_THRESH = 0.10
_IOU_THRESH  = 0.45
_IMG_SIZE    = 640

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - inter
    return inter / union if union else 0

def non_max_suppression_np(predictions, conf_thres=0.25, iou_thres=0.45):
    predictions = predictions[predictions[:, 4] >= conf_thres]
    if len(predictions) == 0:
        return []

    predictions = predictions[np.argsort(-predictions[:, 4])]
    keep = []
    while len(predictions) > 0:
        best = predictions[0]
        keep.append(best)
        if len(predictions) == 1:
            break
        rest = predictions[1:]
        ious = np.array([iou(best[:4], x[:4]) for x in rest])
        predictions = rest[ious < iou_thres]
    return np.stack(keep) if keep else []

def decode_yolo_output(output):
    """
    Convert raw YOLOv8 ONNX output [1, 11, 8400] → [8400, 6]
    Format: [cx, cy, w, h, obj_conf, cls_scores...]
    Returns: [x1, y1, x2, y2, conf, class_id]
    """
    output = output[0]  # shape: (11, 8400)
    output = output.transpose(1, 0)  # shape: (8400, 11)

    boxes = output[:, 0:4]
    obj_conf = output[:, 4:5]
    cls_scores = output[:, 5:]

    class_conf = cls_scores * obj_conf  # shape: (8400, num_classes)
    class_ids = np.argmax(class_conf, axis=1)
    class_scores = np.max(class_conf, axis=1)

    # Convert cx, cy, w, h → x1, y1, x2, y2
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    decoded = np.stack([x1, y1, x2, y2, class_scores, class_ids], axis=1)
    return decoded

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_resized = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img_rgb, (2, 0, 1))[None, :]  # (1, 3, H, W)
    return img_input

def predict_with_onnx(image_path: str) -> str:
    try:
        img_input = preprocess_image(image_path)
        outputs = _SESSION.run(None, {_INPUT_NAME: img_input})

        # Decode raw YOLO output
        pred = decode_yolo_output(outputs[0])

        print("Top 5 class scores (before NMS):")
        top5 = pred[np.argsort(-pred[:, 4])][:5]
        for i, row in enumerate(top5):
            print(f"{i+1}: conf={row[4]:.4f}, class_id={int(row[5])}")
            
        # Apply NMS
        boxes = non_max_suppression_np(pred, conf_thres=_CONF_THRESH, iou_thres=_IOU_THRESH)
        if boxes is None or len(boxes) == 0:
            print("No detections after NMS.")
            return "unknown"

        top_pred = boxes[np.argmax(boxes[:, 4])]
        class_id = int(top_pred[5])

        if class_id < 0 or class_id >= len(_CLASS_NAMES):
            print(f"Invalid class ID: {class_id}")
            return "unknown"

        return _CLASS_NAMES[class_id]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "unknown"

if __name__ == "__main__":
    label = predict_with_onnx("data/test_img.png")  # Make sure this path exists
    print("Predicted mood:", label)
