# python tflite-object-detection.py --modeldir models --graph bestv2-int8.tflite --video videos/Closeupv2.mp4

from tensorflow.lite.python.interpreter import Interpreter
import os
import argparse
import cv2
import numpy as np
import sys
import time


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu


# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH, VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)


# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument


interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print(input_details)
print(output_details)

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname):  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while (video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video!')
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    # if floating_model:
    #     input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[
        0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[0]['index'])[
        0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[
        0]  # Confidence of detected objects

    # # Loop over all detections and draw detection box if confidence is above minimum threshold
    # for i in range(len(scores)):
    #     if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

    #         # Get bounding box coordinates and draw box
    #         # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
    #         ymin = int(max(1, (boxes[i][0] * imH)))
    #         xmin = int(max(1, (boxes[i][1] * imW)))
    #         ymax = int(min(imH, (boxes[i][2] * imH)))
    #         xmax = int(min(imW, (boxes[i][3] * imW)))

    #         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

    #         # # Draw label
    #         # # Look up object name from "labels" array using class index
    #         # object_name = labels[int(classes[i])]
    #         # label = '%s: %d%%' % (object_name, int(
    #         #     scores[i]*100))  # Example: 'person: 72%'
    #         # labelSize, baseLine = cv2.getTextSize(
    #         #     label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
    #         # # Make sure not to draw label too close to top of window
    #         # label_ymin = max(ymin, labelSize[1] + 10)
    #         # # Draw white box to put label text in
    #         # cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (
    #         #     xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
    #         # cv2.putText(frame, label, (xmin, label_ymin-7),
    #         #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text

    # All the results have been drawn on the frame, so it's time to display it.

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    print("FPS: " + fps)

    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
