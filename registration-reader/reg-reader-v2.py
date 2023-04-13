# Import the necessary libraries
import time
from ultralytics import YOLO
import numpy as np
import cv2
from flask import Flask, render_template, Response
from flask import jsonify
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
import threading


# Initialize the Flask web server
app = Flask(__name__)


def ocr_on_bounding_box(image):
    preprocess_text(image)
    text = pytesseract.image_to_string(
        image, config=custom_config)
    print(f"Detected Text: {text}")
    return text


def preprocess_text(img):
    h, w = img.shape[:2]
    # crop left 1/8 - removes false characters from left side
    img = img[:, w//8:]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Apply image smoothing
    thresh = cv2.medianBlur(thresh, 3)

    # Improve the image contrast
    thresh = cv2.equalizeHist(thresh)

    return thresh


@app.route('/')
def index():
    # return 'hellow world'
    return render_template('index.html')


def gen():
    model = YOLO('models/best8v3e10.onnx')  # load a custom model
    videoPath = 'videos/Closeupv2.mp4'

    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        print("Could not open video")
        exit()

    fps_list = []

    # Setting detection zone
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    middle_box_width = 1000
    middle_box_height = 100
    middle_box_x1 = int(frame_width/2 - middle_box_width/2)
    middle_box_y1 = int(frame_height/2 - middle_box_height/2) + 100
    middle_box_x2 = middle_box_x1 + middle_box_width
    middle_box_y2 = middle_box_y1 + middle_box_height

    input_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    time_to_wait = 1.0 / input_frame_rate
    prev_frame_time = 0
    new_frame_time = 0
    fps_pos = (int(frame_width - 150), 30)

    recent_plates = [''] * 5

    crop_size = 640

    # Init 2 frames for threshold
    ret, frame = cap.read()
    ret, frame2 = cap.read()
    height, width, channels = frame.shape

    # plot fps
    time_list = []
    fps_list = []
    total_frames = 0
    first_time = time.time()

    while cap.isOpened():
        start_time = time.time()
        annotated_frame = frame.copy()
        cX = cY = 0

        diff = cv2.absdiff(frame, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        motion_detected = False
        if contours:
            # Get the largest contour (assuming it corresponds to the vehicle)
            largest_contour = contours[0]

            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw a dot at the centroid
                cv2.circle(annotated_frame, (cX, cY), 5, (0, 255, 0), -1)
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)

                if cv2.contourArea(contour) < 900:
                    continue

                if x >= middle_box_x1 and x+w <= middle_box_x2 and y >= middle_box_y1 and y+h <= middle_box_y2:
                    motion_detected = True

        if motion_detected:
            cv2.rectangle(annotated_frame, (middle_box_x1, middle_box_y1),
                          (middle_box_x2, middle_box_y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Status: {}".format('Movement Detected'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
            if cX != 0:
                # Calculate the top-left and bottom-right coordinates of the crop rectangle
                top_left_x = cX - crop_size // 2
                top_left_y = cY - crop_size // 2
                bottom_right_x = cX + crop_size // 2
                bottom_right_y = cY + crop_size // 2

                # Adjust the crop rectangle if it extends past the image boundaries
                if top_left_x < 0:
                    bottom_right_x -= top_left_x
                    top_left_x = 0
                if top_left_y < 0:
                    bottom_right_y -= top_left_y
                    top_left_y = 0
                if bottom_right_x > width:
                    top_left_x -= (bottom_right_x - width)
                    bottom_right_x = width
                if bottom_right_y > height:
                    top_left_y -= (bottom_right_y - height)
                    bottom_right_y = height

                # Crop the image
                frame = frame[top_left_y:bottom_right_y,
                              top_left_x:bottom_right_x]
                # frame = cv2.resize(frame, (crop_size, crop_size))

                # Make a copy of the original frame that is downsized to (320, 320)
                downsized_frame = cv2.resize(frame, (320, 320))

                # Run YOLOv8 inference on the downsized (320, 320) frame
                results = model.predict(
                    downsized_frame, imgsz=320, verbose=False)

                # Get the bounding boxes
                boxes = results[0].boxes

                # Create a new thread for each detected object, translate the bounding box, and run OCR on the original image
                for box in boxes:
                    coords = box.xyxy.numpy()
                    coords = coords[0]
                    coords = [int(coord) * 2 for coord in coords]

                    # annotated_frame = cv2.rectangle(frame, (coords[0], coords[1]), (
                    # coords[2], coords[3]), (0, 255, 0), 2)
                    plate_image = frame[coords[1]
                        :coords[3], coords[0]:coords[2]]
                    # plate_image = cv2.cvtColor(
                    #     plate_image, cv2.COLOR_BGR2GRAY)
                    thread = threading.Thread(
                        target=ocr_on_bounding_box, args=(plate_image,))
                    thread.start()

        else:
            cv2.rectangle(annotated_frame, (middle_box_x1, middle_box_y1),
                          (middle_box_x2, middle_box_y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, "Status: {}".format('No Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        for i, plate in enumerate(recent_plates):
            cv2.putText(annotated_frame, plate, (10, frame_height - 30 * (5 - i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps_list.append(fps)
        fps = str(fps)
        time_list.append(new_frame_time-first_time)
        cv2.putText(annotated_frame, "FPS: {}".format(fps), fps_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        # print("FPS: " + fps)
        processing_time = time.time() - start_time
        # adjusted_wait_time = max(time_to_wait - processing_time, 0)
        # time.sleep(adjusted_wait_time)

        out = cv2.imencode('.jpg', annotated_frame)[1].tobytes()

        frame = frame2
        ret, frame2 = cap.read()
        if not ret:
            break
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #     ret, frame = cap.read()
        #     ret, frame2 = cap.read()
        # assert not isinstance(frame, type(None)), 'frame not found'

        # Yield the frame to the Flask site
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + out + b'\r\n')
    plt.figure()
    plt.plot(time_list, fps_list)
    plt.xlabel('Time')
    plt.ylabel('FPS')
    # save the figure
    plt.savefig('PiFPS-DayVideo.png')


@ app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Run the Flask web server
if __name__ == '__main__':
    whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
    custom_config = r'--psm 6 -c tessedit_char_whitelist={}'.format(whitelist)
    app.run(port=8000, debug=True)
