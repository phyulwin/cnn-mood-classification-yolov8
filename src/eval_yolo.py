#!/usr/bin/env python3
"""
eval_yolo.py
Evaluate YOLOv8 model, plot training metrics, run validation, and export ONNX.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

def plot_training():
    RUN_DIR     = "runs/train/yolo_mood"
    metrics_csv = os.path.join(RUN_DIR, "results.csv")

    # 1) Load metrics CSV
    try:
        df = pd.read_csv(metrics_csv)
    except Exception as e:
        print(f"Error reading {metrics_csv}: {e}")
        return

    # 2) Debug: show actual columns
    print("Columns in results.csv:", df.columns.tolist())

    # 3) Rename to simpler names
    rename_map = {
        'train/box_loss':        'box_loss',
        'train/cls_loss':        'cls_loss',
        'train/dfl_loss':        'dfl_loss',
        'metrics/precision(B)':  'precision',
        'metrics/mAP50(B)':      'map50',
        # optional if you want recall or mAP50-95
        'metrics/recall(B)':     'recall',
        'metrics/mAP50-95(B)':   'map50_95'
    }
    df = df.rename(columns=rename_map)

    # 4) Plot training losses
    plt.figure(figsize=(10,5))
    plt.plot(df['epoch'], df['box_loss'], label="box_loss")
    plt.plot(df['epoch'], df['cls_loss'], label="cls_loss")
    plt.plot(df['epoch'], df['dfl_loss'], label="dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("YOLOv8 Training Losses")
    plt.savefig(os.path.join(RUN_DIR, "losses.png"))
    plt.close()

    # 5) Plot precision & mAP
    plt.figure(figsize=(10,5))
    plt.plot(df['epoch'], df['precision'], label="precision")
    plt.plot(df['epoch'], df['map50'],    label="mAP@0.5")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("YOLOv8 Precision & mAP")
    plt.savefig(os.path.join(RUN_DIR, "precision_map.png"))
    plt.close()

    print("Saved training plots in", RUN_DIR)

def validate_and_export():
    DATA_CFG = "data/mood_data.yaml"
    RUN_DIR  = "runs/train/yolo_mood"
    VAL_DIR  = "runs/val/yolo_mood_val"
    IMG_SIZE = 640

    os.makedirs(VAL_DIR, exist_ok=True)

    # 1) Validation
    try:
        model = YOLO(os.path.join(RUN_DIR, "weights", "best.pt"))
        model.val(data=DATA_CFG, project="runs/val", name="yolo_mood_val")
        print("Validation complete. Results in", VAL_DIR)
    except Exception as e:
        print(f"Validation error: {e}")

    # 2) Try YOLOv8’s built-in ONNX exporter
    try:
        exporter = YOLO(os.path.join(RUN_DIR, "weights", "best.pt"))
        exporter.export(format="onnx", imgsz=IMG_SIZE)
        print("Exported best model to ONNX via YOLOv8 exporter.")
        return
    except Exception as e:
        print(f"YOLO ONNX export failed: {e}")

    # 3) Fallback: manual torch.onnx.export
    try:
        import torch
        yolo = YOLO(os.path.join(RUN_DIR, "weights", "best.pt"))
        net = yolo.model
        net.eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        onnx_path = os.path.join(RUN_DIR, "weights", "best_manual.onnx")
        torch.onnx.export(
            net,
            dummy,
            onnx_path,
            opset_version=12,
            input_names=['images'],
            output_names=['output']
        )
        print("Manual ONNX export succeeded:", onnx_path)
    except Exception as e2:
        print(f"Manual ONNX export also failed: {e2}")

def main():
    plot_training()
    validate_and_export()

if __name__ == "__main__":
    main()
