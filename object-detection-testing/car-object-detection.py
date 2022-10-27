# Testing openCV object detection on a video file with CVLib

import numpy as np
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
import io
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from pydantic import BaseModel
import pafy


URL = "https://www.youtube.com/watch?v=wqctLW0Hb_0&t=132s" 
play = pafy.new(URL).streams[-1] #'-1' means read the lowest quality of video.
assert play is not None 
stream = cv2.VideoCapture(play.url) #create a opencv video stream.


# app = FastAPI(__name__)
# class ImageType(BaseModel):
#  url: str
# @app.get(“/”)
# def home():
#  return “Home”
# @app.post(“/predict/”) 
# def prediction(request: Request, 
#  file: bytes = File(…)):
# if request.method == “POST”:
#     image_stream = io.BytesIO(file)
#     image_stream.seek(0)
#     file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
#     frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     bbox, label, conf = cv.detect_common_objects(frame)
#     output_image = draw_bbox(frame, bbox, label, conf)
#     num_cars = label.count(‘car’)
#     print(‘Number of cars in the image is ‘+ str(num_cars))
#     return {“num_cars”:num_cars}
#  return “No post request found”
