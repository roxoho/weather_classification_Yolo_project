from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt') 

results = model.train(data='D:\weather_classification_project\data', epochs=20, imgsz=64)