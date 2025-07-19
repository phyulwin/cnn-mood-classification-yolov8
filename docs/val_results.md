# Validation Results

Below are the validation outputs from `runs/val/yolo_mood_val/`.

---

## Box Precision Curve  
![](../runs/val/yolo_mood_val/BoxP_curve.png)  
*Precision of bounding‑box predictions vs. validation epochs. Higher is better—shows how accurately the model localizes faces over time.*

## Box Recall Curve  
![](../runs/val/yolo_mood_val/BoxR_curve.png)  
*Recall of bounding‑box predictions vs. validation epochs. Indicates the model’s ability to find all faces.*

## Box F1‑Score Curve  
![](../runs/val/yolo_mood_val/BoxF1_curve.png)  
*Harmonic mean of precision and recall vs. validation epochs. Balances detection accuracy and completeness.*

## Box Precision–Recall Curve  
![](../runs/val/yolo_mood_val/BoxPR_curve.png)  
*Precision–Recall trade‑off for face detection on the validation set. Illustrates performance at different confidence thresholds.*

---

## Confusion Matrices

### Raw Confusion Matrix  
![](../runs/val/yolo_mood_val/confusion_matrix.png)  
*Counts of predicted mood labels vs. true labels. Diagonal entries are correct predictions.*

### Normalized Confusion Matrix  
![](../runs/val/yolo_mood_val/confusion_matrix_normalized.png)  
*Percentages of predictions per true class. Helps spot systematic biases or misclassifications.*

---

## Sample Batch Visualizations

### Batch 0 – Ground Truth  
![](../runs/val/yolo_mood_val/val_batch0_labels.jpg)  
*Validation images with ground‑truth bounding boxes and mood labels.*

### Batch 0 – Predictions  
![](../runs/val/yolo_mood_val/val_batch0_pred.jpg)  
*Same images with the model’s predicted boxes and labels for comparison.*

---

### Batch 1 – Ground Truth  
![](../runs/val/yolo_mood_val/val_batch1_labels.jpg)  
*Validation images with ground‑truth boxes and labels.*

### Batch 1 – Predictions  
![](../runs/val/yolo_mood_val/val_batch1_pred.jpg)  
*Model’s predictions on batch 1 for visual inspection.*

---

### Batch 2 – Ground Truth  
![](../runs/val/yolo_mood_val/val_batch2_labels.jpg)  
*Validation images with ground‑truth boxes and labels.*

### Batch 2 – Predictions  
![](../runs/val/yolo_mood_val/val_batch2_pred.jpg)  
*Model’s predicted boxes and labels on batch 2.*

---