### This project is based on the following research paper:  
**[Building of a Convolutional Neuronal Network for the prediction of mood states through face recognition based on object detection with YOLOV8 and Python](https://ieeexplore.ieee.org/document/10372862)**

## Goal of Research Paper

Develop a real‑time CNN model to predict six mood states from facial expressions using object detection and pattern recognition:

- Tension
- Frustration
- Anxiety
- Fatigue
- Neutrality
- Happiness

## Data used in the Research

- Custom dataset of **2,500 labeled face images** (~420 per class, 400 for fatigue)
- Images standardized and augmented (resolution, lighting ±15%) to **4,000–4,500 examples**
- Split: **80% train**, **10% validation**, **10% test**

## Tools & Frameworks

- Python (Anaconda)
- YOLOv8 for face detection
- Roboflow for dataset annotation
- Google Colab Jupyter for training
- 1080p HD webcam for real‑time inference

## Key Metrics

- Consistently high **mAP of 99.5%** and ~**99% precision** across incremental CNN versions
- YOLOv8 trained for **155 epochs**
  - Best precision: **94.3% at epoch 129**
  - Precision: **93.9% at epoch 155**

---

## Complete Citation & Abstract

```
F. E. Ramirez Rios and A. María Reyes Duke, "Building of a Convolutional Neuronal Network for the prediction of mood states through face recognition based on object detection with YOLOV8 and Python," 2023 IEEE International Conference on Machine Learning and Applied Network Technologies (ICMLANT), San Salvador, El Salvador, 2023, pp. 1-6, doi: 10.1109/ICMLANT59547.2023.10372862.

Abstract: This research focuses on developing a real-time mood state prediction CNN model from the extraction of facial expressions. It targets 6 mood states (Tension, Frustration, Anxiety, Fatigue, Neutrality, Happiness) via facial feature extraction and pattern recognition. The project begins with the creation of the dataset which is later preprocessed to standardize image resolutions and lighting conditions to generate additional training examples, augmenting the images from 2500 to 4000 for training. Adopting the agile incremental development approach to train the CNN, it is ensured that each iteration of the model is fully functional and builds upon the model’s previous versions, using them as checkpoints, to focus on the maximization of the precision and mAP value, maintaining a mAP value of 99.5% in each iteration with little to no fluctuation on the 99% precision value on every version of the model. The research then transitions to manual training using the YOLOv8 object detection framework on Google Colab Jupyter notebook service. The YOLOv8 model achieved consistent high precision during its 155 epoch training, obtaining the model’s best weight at epoch 129, with a precision of 94.3%. This result indicates the potential of the model for the development of systems that respond to human feelings effectively.

Keywords: Training; Mood; Face recognition; Computational modeling; Object detection; Manuals; Predictive models; Convolutional Neural Networks; Deep Learning; Facial Detection; Facial Features; Mood States

URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10372862&isnumber=10372861
```