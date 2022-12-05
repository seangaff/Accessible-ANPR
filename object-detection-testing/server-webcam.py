# Same as webcam-test.py but changed to display output to a simple Flask site.

# Import the necessary libraries
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from flask import Flask, render_template, Response

# Initialize the Flask web server
app = Flask(__name__)

# Create a route that will display the live video stream on the Flask site


@app.route('/')
def index():
    # return 'hellow world'
    return render_template('index.html')


def gen():
    # Open the webcam using OpenCV
    cap = cv2.VideoCapture(0)

    # Keep looping indefinitely
    while True:
        # Grab the next frame from the webcam
        status, frame = cap.read()

        if not status:
            print("Could not read frame")
            exit()

        bbox, label, conf = cv.detect_common_objects(
            frame, confidence=0.25, model='yolov4-tiny')    # YoloV4-tiny is used for object detection

        # draw bounding box over detected objects
        out = draw_bbox(frame, bbox, label, conf)

        # Convert the frame to JPEG format
        out = cv2.imencode('.jpg', out)[1].tobytes()

        # Yield the frame to the Flask site
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + out + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Run the Flask web server
if __name__ == '__main__':
    app.run(port=8000, debug=True)
