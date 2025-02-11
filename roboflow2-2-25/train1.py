import os
from ultralytics import YOLO

# 🔹 Define dataset paths (change these based on your dataset structure)
dataset_path = "/DeerGuard"  # Update this
train_json = os.path.join(dataset_path, "train/annotations.json")
val_json = os.path.join(dataset_path, "val/annotations.json")
test_json = os.path.join(dataset_path, "test/annotations.json")

# 🔹 Create YAML config file for YOLOv8 training
dataset_yaml = "data.yaml"

yaml_content = f"""
path: {dataset_path}  # Root dataset directory
train: train/images  # Train images directory (relative to dataset_path)
val: val/images      # Validation images directory (relative to dataset_path)
test: test/images    # Test images directory (relative to dataset_path)
names:
  0: 'class_1'
  1: 'class_2'
  2: 'class_3'  # Modify according to your dataset's classes
"""

# 🔹 Save dataset YAML file
with open(dataset_yaml, "w") as f:
    f.write(yaml_content)

# 🔹 Load YOLOv8 model (pretrained on COCO or custom)
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt', 'yolov8m.pt', etc.

# 🔹 Train the YOLOv8 model
model.train(
    data=dataset_yaml,  # Path to dataset.yaml
    epochs=50,          # Number of training epochs
    batch=16,           # Batch size (adjust based on GPU memory)
    imgsz=640,         # Image size for training
    workers=4,         # Number of data loader workers
    device="cuda"      # Use GPU ('cuda') or CPU ('cpu')
)

# 🔹 Evaluate the trained model on test data
metrics = model.val()

# 🔹 Save/export trained model (optional)
model.export(format="onnx")  # Export to ONNX format
