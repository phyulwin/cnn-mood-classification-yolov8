# Training Summary

**Epochs Completed:** 155  
**Total Time:** 2.832 hours

**Optimizer Stripped From:**
- `runs/train/yolo_mood6/weights/last.pt` (6.3 MB)
- `runs/train/yolo_mood6/weights/best.pt` (6.3 MB)

---

# Validation Output

```
Validating runs/train/yolo_mood6/weights/best.pt...
Ultralytics 8.3.168 | Python 3.13.5 | torch 2.7.1+cu118
CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB)
Model summary (fused): 72 layers, 3,007,013 params, 8.1 GFLOPs
```

---

## Speed

- **Preprocess:** 0.2 ms/image
- **Inference:** 1.9 ms/image
- **Loss:** 0.0 ms/image
- **Postprocess:** 1.4 ms/image

---

# Model Output

- **Results Directory:** `runs/train/yolo_mood6`
- **Best Weights:** `runs/train/yolo_mood/weights/best.pt`
- **Last Weights:** `runs/train/yolo_mood/weights/last.pt`

---

# ONNX Export Summary

**ONNX Version:** 1.18.0  
**Opset:** 19

**Export Status:** Success  
**Duration:** 299.5 s  
**Output File:** `runs/train/yolo_mood/weights/best.onnx` (11.7 MB)  
**Total Time:** 299.8 s

> **Warning:** Simplifier failed: missing module `onnxslim`.

---

## Visualize Model Graph

Open the `.onnx` file in [Netron](https://netron.app).