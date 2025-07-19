## Project Overview

### Data Conversion (`convert_to_yolo.py`)
- Converts a raw folder of class-labeled face images (`data/raw/.../1, 2, … 7`) into YOLO format.
- Assigns a full-image bounding box to each image.
- Writes label IDs to `.txt` files in YOLO format.
- Outputs processed images and labels to `data/processed/images` and `data/processed/labels`.

### Subset Sampling (`sample_subset.py`)
- Randomly selects 6,000 image-label pairs from the full YOLO dataset.
- Enables faster training and experimentation.

### Preprocessing & Loading (`preprocess.py`)
- Prepares image data using PyTorch's `ImageFolder` and `DataLoader`.
- Produces normalized 224×224 tensor batches (for testing or custom CNNs).

### Model Training (`train_yolo.py`)
- Fine-tunes the YOLOv8n model for classification on full-frame face images.
- Uses `data/mood_data.yaml` to configure classes and paths.
- Runs for 155 epochs with 640×640 input size.
- Saves model checkpoints and metrics to `runs/train/yolo_mood/`.

### Evaluation & Export (`eval_yolo.py`)
- Loads the best checkpoint and validates on the test set.
- Generates performance plots (loss, mAP, precision).
- Exports the final model to ONNX format for deployment/interoperability.

### GPU Check (`use_gpu.py`)
- Verifies CUDA device availability and prints GPU name.

---

## Project Highlights

### Single-pass YOLO Classification
- No secondary CNN needed—mood labels are treated as detection classes directly.

### Result Tracking
- Evaluation metrics (box_loss, mAP@0.5, precision) are logged and visualized.

### ONNX Export Support
- Model can be exported for integration into non-PyTorch environments.

### Subset Sampling for Speed
- Enables fast prototyping on a smaller image set.