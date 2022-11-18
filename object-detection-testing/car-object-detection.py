# Testing openCV object detection on a video file with CVLib
# https://www.cvlib.net/
# YoloV4 is used for object detection
# COCO dataset


import io
import time
import numpy as np
import sys
import os
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
# from IPython.display import display
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow debugging logs

cap = cv2.VideoCapture(
    'road-traffic-sample.mp4')
# fps = int(cap.get(cv2.CAP_PROP_FPS))
if not cap.isOpened():
    print("Could not open video")
    exit()

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while cap.isOpened():
    status, frame = cap.read()

    # Our operations on the frame come here
    # gray = frame

    # # resizing the frame size according to our need
    # gray = cv2.resize(gray, (500, 300))

    # Calculating the fps
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # bbox, label, conf = cv.detect_common_objects(
    #     gray, confidence=0.25, model='yolov4-tiny')    # YoloV4-tiny is used for object detection
    bbox, label, conf = cv.detect_common_objects(
        frame, confidence=0.25)
    print(bbox, label, conf)
    out = draw_bbox(frame, bbox, label, conf)
    cv2.imshow("Real-time object detection", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
