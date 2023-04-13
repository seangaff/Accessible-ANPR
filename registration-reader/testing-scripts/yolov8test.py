import cv2
import pytesseract
from ultralytics import YOLO
import threading
import easyocr
import numpy as np
import time


def ocr_on_bounding_box(image):
    text = pytesseract.image_to_string(image)
    # ocr_result = reader.readtext(np.asarray(
    #     cropped_image), allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    # if ocr_result:
    #     text = ocr_result[0][1]
    print(f"Detected Text: {text}")


def preprocess_text(img):
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

    # Deskew the image
    coords = cv2.findNonZero(thresh)
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    thresh = cv2.warpAffine(
        thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return thresh

    # Load the YOLOv8 model
model = YOLO('models/best8v3e10.onnx')
# reader = easyocr.Reader(['en'], gpu=False)

# Open the video file
video_path = "videos/Closeupv2.mp4"
cap = cv2.VideoCapture(video_path)

# Create a MOG2 background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=100, detectShadows=False)

crop_size = 640

input_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
time_to_wait = 1.0 / input_frame_rate
prev_frame_time = 0
new_frame_time = 0
fps_pos = (0, 30)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        start_time = time.time()

        height, width, channels = frame.shape

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Remove noise using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if contours:
            # Get the largest contour (assuming it corresponds to the vehicle)
            largest_contour = contours[0]

            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw a dot at the centroid
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

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
                frame = cv2.resize(frame, (crop_size, crop_size))

            # Make a copy of the original frame that is downsized to (320, 320)
            downsized_frame = cv2.resize(frame, (320, 320))

            # Run YOLOv8 inference on the downsized (320, 320) frame
            results = model.predict(downsized_frame, imgsz=320)

            # Get the bounding boxes
            boxes = results[0].boxes

            # Create a new thread for each detected object, translate the bounding box, and run OCR on the original image
            for box in boxes:
                coords = box.xyxy.numpy()
                coords = coords[0]
                coords = [int(coord) * 2 for coord in coords]

                annotated_frame = cv2.rectangle(frame, (coords[0], coords[1]), (
                    coords[2], coords[3]), (0, 255, 0), 2)
                # print(coords)
                cropped_image = frame[coords[1]:coords[3], coords[0]:coords[2]]
                # print(cropped_image.shape)
                cropped_image = preprocess_text(cropped_image)
                thread = threading.Thread(
                    target=ocr_on_bounding_box, args=(cropped_image,))
                thread.start()

                # Display the annotated frame
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:

                    cv2.imshow("YOLOv8 Inference", cropped_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()