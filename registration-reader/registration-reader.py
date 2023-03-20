# Import the necessary libraries
import time
import os
import numpy as np
import cv2
from flask import Flask, render_template, Response
from flask import jsonify
import torch
import easyocr
import argparse
import matplotlib.pyplot as plt

# Initialize the Flask web server
app = Flask(__name__)

# Create a route that will display the live video stream on the Flask site


@app.route('/')
def index():
    # return 'hellow world'
    return render_template('index.html')


def get_plates_xy(frame: np.ndarray, labels: list, row: list, width: int, height: int, reader: easyocr.Reader) -> tuple:
    '''Get the results from easyOCR for each frame and return them with bounding box coordinates'''

    x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]
                                                                * width), int(row[3]*height)  # BBOx coordniates
    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BBox
    # , paragraph="True", min_size=50)
    ocr_result = reader.readtext(np.asarray(
        plate_crop), allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    # print(ocr_result)
    return ocr_result, x1, y1


def detect_text(i: int, row: list, x1: int, y1: int, ocr_result: list, detections: list, yolo_detection_prob: float = 0.3) -> list:
    '''Checks the detection's probability, discards those with low prob and rewrites output from ocr_reader to >>detections<< list'''

    if row[4] >= yolo_detection_prob:  # discard predictions below the value
        if (len(ocr_result)) > 0:
            for item in ocr_result:
                detections[i][0] = item[1]
                detections[i][1] = [x1, y1]
                detections[i][2] = item[2]

    return detections


def is_adjacent(coord1: list, coord2: list) -> bool:
    '''Checks if [x, y] from list coord1 is similar to coord2'''

    MAX_PIXELS_DIFF = 50

    if (abs(coord1[0] - coord2[0]) <= MAX_PIXELS_DIFF) and (abs(coord1[1] - coord2[1]) <= MAX_PIXELS_DIFF):
        return True
    else:
        return False


def sort_detections(detections: list, plates_data: list) -> list:
    '''Looks at detections from last frame and rewrites indexes for similar coordinates'''

    for m in range(0, len(detections)):
        for n in range(0, len(plates_data)):
            if not detections[m][1] == [0, 0] and not plates_data[n][1] == [0, 0]:
                if is_adjacent(detections[m][1], plates_data[n][1]):
                    if m != n:
                        temp = detections[m]
                        detections[m] = detections[n]
                        detections[n] = temp

    return detections


def delete_old_labels(detections: list, count_empty_labels: list, plates_data: list, frames_to_reset: int = 3) -> tuple:
    '''If earlier detected plate isn't spotted for the next >>FRAMES_TO_RESET<< frames, delete it from >>plates_data<<'''

    for m in range(0, len(detections)):
        if detections[m][0] == 'None' and not count_empty_labels[m] == frames_to_reset:
            count_empty_labels[m] += 1
        elif count_empty_labels[m] == frames_to_reset:
            count_empty_labels[m] = 0
            plates_data[m] = ['None', [0, 0], 0]
        else:
            count_empty_labels[m] = 0

    return plates_data, count_empty_labels


def overwrite_plates_data(i, detections: list, plates_data: list, plate_lenght=None) -> list:
    '''Checks coordinates from >>detections<<, if there is similar record in >>plate_data<< tries to overwrite it (only if probability is higher)'''

    if (detections[i][2] > plates_data[i][2] or detections[i][2] == 0):
        if plate_lenght:
            if len(detections[i][0]) == plate_lenght:
                plates_data[i][0] = detections[i][0]
                plates_data[i][2] = detections[i][2]
        else:
            plates_data[i][0] = detections[i][0]
            plates_data[i][2] = detections[i][2]
    plates_data[i][1] = detections[i][1]

    return plates_data


def gen():
    videoPath = 'videos/CloseupMini.mp4'
    cap = cv2.VideoCapture(videoPath)
    # cap = cv2.VideoCapture(
    #     "http://192.168.0.122:81")

    if not cap.isOpened():
        print("Could not open video")
        exit()

    fps_list = []

    # Setting detection zone
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    middle_box_width = 800
    middle_box_height = 100
    middle_box_x1 = int(frame_width/2 - middle_box_width/2)
    middle_box_y1 = int(frame_height/2 - middle_box_height/2) + 100
    middle_box_x2 = middle_box_x1 + middle_box_width
    middle_box_y2 = middle_box_y1 + middle_box_height

    plates_data = [['None', [0, 0], 0] for n in range(5)]
    count_empty_labels = [0]*5

    input_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    time_to_wait = 1.0 / input_frame_rate
    prev_frame_time = 0
    new_frame_time = 0
    fps_pos = (int(frame_width - 150), 30)

    # Init 2 frames for threshold
    ret, frame = cap.read()
    ret, frame2 = cap.read()

    recent_plates = [''] * 5

    while cap.isOpened():
        start_time = time.time()
        # ret, frame = cap.read()

        diff = cv2.absdiff(frame, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 900:
                continue

            if x >= middle_box_x1 and x+w <= middle_box_x2 and y >= middle_box_y1 and y+h <= middle_box_y2:
                motion_detected = True

        if motion_detected:
            results = model(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            labels, coordinates = results.xyxyn[0][:, -
                                                   1], results.xyxyn[0][:, :-1]

            detections = [['None', [0, 0], 0] for n in range(5)]
            i = 0

            # Read all detected plates per each frame and save them to >>detections<<
            while i < len(labels):
                row = coordinates[i]
                # 3. Crop detections and pass them to the easyOCR
                ocr_result, x1, y1 = get_plates_xy(
                    frame, labels, row, frame_width, frame_height, reader)

                # 4. Get reading for the each frame
                detections = detect_text(
                    i, row, x1, y1, ocr_result, detections, 0.5)
                i += 1
            i = 0

            # 5. Do some tracking and data managing for better results
            # If we get multiple detections in one frame easyOCR mixes them every few frames, so here we make sure that they are saved according to the \
            # detections' coordinates. Then we delete data about plates that dissapeared for more than >>frames_to_reset<< frames. And finally we overwrite \
            # the predictions (regarding to the probability of easyOCR detections - if new predcition has less p% than the previous one, we skip it.)

            # Sort detections
            detections = sort_detections(detections, plates_data)

            # Delete data about plates that dissapeared from frame
            plates_data, count_empty_labels = delete_old_labels(
                detections, count_empty_labels, plates_data, 3)

            # Overwrite data and print text predictions over the boxes
            while i < len(labels):
                plates_data = overwrite_plates_data(
                    i, detections, plates_data, 7)
                if plates_data[i][0] not in recent_plates and plates_data[i][0] != 'None':
                    recent_plates.pop(0)
                    recent_plates.append(plates_data[i][0])
                cv2.putText(frame, f"{plates_data[i][0]}", (plates_data[i][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                i += 1
            cv2.rectangle(frame, (middle_box_x1, middle_box_y1),
                          (middle_box_x2, middle_box_y2), (0, 255, 0), 2)
            cv2.putText(frame, "Status: {}".format('Movement Detected'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        else:
            cv2.rectangle(frame, (middle_box_x1, middle_box_y1),
                          (middle_box_x2, middle_box_y2), (0, 0, 255), 2)
            cv2.putText(frame, "Status: {}".format('No Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        for i, plate in enumerate(recent_plates):
            cv2.putText(frame, plate, (10, frame_height - 30 * (5 - i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps_list.append(fps)
        fps = str(fps)
        cv2.putText(frame, "FPS: {}".format(fps), fps_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        print("FPS: " + fps)

        out = cv2.imencode('.jpg', frame)[1].tobytes()

        frame = frame2
        ret, frame2 = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            ret, frame2 = cap.read()
        assert not isinstance(frame, type(None)), 'frame not found'

        processing_time = time.time() - start_time
        adjusted_wait_time = max(time_to_wait - processing_time, 0)
        time.sleep(adjusted_wait_time)

        # Yield the frame to the Flask site
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + out + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Run the Flask web server
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example script that takes command line arguments')

    parser.add_argument('--arg1', type=str, default='default_value',
                        help='Description of argument 1')
    parser.add_argument('--arg2', type=int, default=42,
                        help='Description of argument 2')

    # Parse the arguments
    args = parser.parse_args()

    # Set the global variables based on the command line arguments
    global_var1 = args.arg1
    global_var2 = args.arg2

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='models/bestv2.onnx', force_reload=True)
    reader = easyocr.Reader(['en'], gpu=False)
    app.run(port=8000, debug=True)
