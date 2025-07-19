# Training Results

Below are the artifacts generated during the YOLOv8 training run in `runs/train/yolo_mood/`.

---

## Summary Plot

![](../runs/train/yolo_mood/results.png)  
*Combined view of key metrics (loss, precision, recall, mAP) over all epochs.*

---

## Loss & Score Curves

![](../runs/train/yolo_mood/losses.png)  
*Box, classification, and DFL losses vs. epoch.*

![](../runs/train/yolo_mood/precision_map.png)  
*Precision and mAP@0.5 vs. epoch.*

---

## Bounding‑Box Metrics

![](../runs/train/yolo_mood/BoxP_curve.png)  
*Box Precision vs. epoch.*

![](../runs/train/yolo_mood/BoxR_curve.png)  
*Box Recall vs. epoch.*

![](../runs/train/yolo_mood/BoxF1_curve.png)  
*Box F1‑Score vs. epoch.*

![](../runs/train/yolo_mood/BoxPR_curve.png)  
*Precision–Recall curve for bounding‑boxes.*

---

## Confusion Matrices

![](../runs/train/yolo_mood/confusion_matrix.png)  
*Raw counts of predicted vs. true classes.*

![](../runs/train/yolo_mood/confusion_matrix_normalized.png)  
*Normalized (percentage) confusion matrix.*

---

## Sample Training Batches

### Early Batches  
![](../runs/train/yolo_mood/train_batch0.jpg)  
![](../runs/train/yolo_mood/train_batch1.jpg)  
![](../runs/train/yolo_mood/train_batch2.jpg)  

### Later Batches  
![](../runs/train/yolo_mood/train_batch27260.jpg)  
![](../runs/train/yolo_mood/train_batch27261.jpg)  
![](../runs/train/yolo_mood/train_batch27262.jpg)  

---

## Validation Snapshots (during training)

![](../runs/train/yolo_mood/val_batch0_labels.jpg)  
![](../runs/train/yolo_mood/val_batch0_pred.jpg)  

![](../runs/train/yolo_mood/val_batch1_labels.jpg)  
![](../runs/train/yolo_mood/val_batch1_pred.jpg)  

![](../runs/train/yolo_mood/val_batch2_labels.jpg)  
![](../runs/train/yolo_mood/val_batch2_pred.jpg)  

---

## Configuration

- **args.yaml**  
  Contains the training hyperparameters and settings used for this run.

---

## Raw Metrics

- **results.csv**  
  Epoch‑wise metrics (losses, precision, recall, mAP, learning rates).

---