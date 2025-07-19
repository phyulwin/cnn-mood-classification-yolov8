# . Update YOLO’s “names” so everywhere uses the right labels

from ultralytics import YOLO

model = YOLO("runs/train/yolo_mood/weights/best.pt")

# Replace the seven wrong labels with your target moods
model.names = [
    "Tension",     # was “surprised”
    "Frustration", # was “fear”
    "Anxiety",     # was “disgust”
    "Fatigue",     # was “smug”
    "Neutrality",  # was “sad”
    "Happiness",   # was “angry”
    "Neutrality"   # was “neutral”  ← you may want to merge or pick one “neutral”
]

# Now any .predict or .export will carry these new names
results = model.predict("some.jpg")
print(results.names)  # your new list
model.export(format="onnx") 
