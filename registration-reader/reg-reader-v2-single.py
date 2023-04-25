import cv2
import pytesseract
from pytesseract import Output
from ultralytics import YOLO
import numpy as np
import os


def ocr_on_bounding_box(image):
    image = preprocess_text(image)
    # cv2.imshow("YOLOv8 Inference", image)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(
        image, config=custom_config)
    print(f"Detected Text: {text}")
    return text


def preprocess_text(img):
    h, w = img.shape[:2]
    # crop left 1/8 - removes false characters from left side
    img = img[:, w//10:]
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


with open('readerOutput.txt', 'w') as f:
    f.write("")

model = YOLO('models/best8v3e10.onnx')

whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
custom_config = r'--psm 6 -c tessedit_char_whitelist={}'.format(whitelist)
# get img shape
line_number = 1

for filename in os.listdir('images'):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img = cv2.imread(os.path.join('images', filename))

        img_shape = img.shape
        orig_height, orig_width = img_shape[0], img_shape[1]

        # Calculate scaling factors for width and height
        width_scaling_factor = orig_width / 320
        height_scaling_factor = orig_height / 320

        downsized_frame = cv2.resize(img, (320, 320))
        cv2.imshow("YOLOv8 Inference", downsized_frame)
        cv2.waitKey(0)
        break

        # Run YOLOv8 inference on the downsized (320, 320) frame
        results = model.predict(downsized_frame, imgsz=320)

        # Get the bounding boxes
        boxes = results[0].boxes

        # Create a new thread for each detected object, translate the bounding box, and run OCR on the original image
        for box in boxes:
            coords = box.xyxy.numpy()
            coords = coords[0]
            coords = [int(round(coord * width_scaling_factor)) if idx % 2 == 0 else int(
                round(coord * height_scaling_factor)) for idx, coord in enumerate(coords)]
            cropped_image = img[coords[1]:coords[3], coords[0]:coords[2]]
            # cv2.imshow("YOLOv8 Inference", cropped_image)
            # processed = preprocess_text(cropped_image)
            dims = cropped_image.shape
            cropped_image = cv2.resize(
                cropped_image, (dims[1] * 2, dims[0] * 2))
            text = ocr_on_bounding_box(cropped_image)
            text = text.strip()
            with open('readerOutput.txt', 'a') as f:
                f.write(f"{line_number}. {text}\n")

            # Increment the line number
            line_number += 1
            cv2.putText(img, "{}".format(text), (coords[0], coords[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.rectangle(img, (coords[0], coords[1]), (
                coords[2], coords[3]), (0, 255, 0), 2)
            cv2.imshow("YOLOv8 Inference", img)
            cv2.waitKey(0)
    break
