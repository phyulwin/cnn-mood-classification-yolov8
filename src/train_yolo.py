#!/usr/bin/env python3
"""
train_yolo.py
Train YOLOv8 model for mood state prediction using Ultralytics YOLOv8.
"""

from ultralytics import YOLO
import os
import torch 

def main():
    DATA_CFG = "data/mood_data.yaml"
    EPOCHS   = 155
    IMG_SIZE = 640
    BATCH    = 32
    PROJECT  = "runs/train"
    NAME     = "yolo_mood"

    os.makedirs(os.path.join(PROJECT, NAME, "weights"), exist_ok=True)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=DATA_CFG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        save=True,
        cache=True,
        device='cuda'
    )

    print("Training complete.")
    print(f"Best weights: {PROJECT}/{NAME}/weights/best.pt")
    print(f"Last weights: {PROJECT}/{NAME}/weights/last.pt")

if __name__ == "__main__":
    main()
