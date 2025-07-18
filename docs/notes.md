notes of the research paper

Goal: Develop a real‑time CNN model to predict six mood states (Tension, Frustration, Anxiety, Fatigue, Neutrality, Happiness) from facial expressions via object detection and pattern recognition 

Data: Custom dataset of 2,500 labeled face images (≈420 per class, 400 for fatigue), standardized and augmented (resolution, lighting ±15%) to 4,000–4,500 examples; split 80% train, 10% validation, 10% test 

Tools/Frameworks: Python (Anaconda), YOLOv8 for face detection, Roboflow for dataset annotation, Google Colab Jupyter for training, and a 1080p HD webcam for real‑time inference 

Key Metrics: Consistently high mAP of 99.5% and ~99% precision across incremental CNN versions; YOLOv8 trained for 155 epochs, with best precision 94.3% at epoch 129 and 93.9% at epoch 155 


Activate (so you see (venv) in your prompt):
```.\venv\Scripts\Activate.ps1```

Install your requirements:
```pip install -r requirements.txt```

Verify they landed in the venv:
```pip list```

Deactivate the virtual env
In your PowerShell prompt (where you see (venv)), just run:
```deactivate```