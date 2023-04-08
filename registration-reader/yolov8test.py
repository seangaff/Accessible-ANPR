# from ultralytics import YOLO

# # model = YOLO('yolov8n.pt')
# model = YOLO('models/best8v1e6.pt')  # load a custom model
# results = model.track(source="videos/IrelandCompliation.mp4",
#                       conf=0.3, iou=0.5, show=True)


import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')
model = YOLO('models/best8v1e6.pt')

# Open the video file
image = "videos/car.jpeg"

result = model(image)
