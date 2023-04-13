import cv2
import numpy as np
import easyocr
# from PIL import Image
import cv2
import time


def preprocess_image(image, target_size):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(
        image_normalized, (2, 0, 1))  # Change to NCHW format
    image_expanded = np.expand_dims(image_transposed, axis=0)
    # return image_expanded.astype(np.float16)
    return image_expanded


def draw_bounding_boxes(image, boxes, box_format, confidence_threshold=0.5):
    for box in boxes:
        if box_format == "xywh":
            x, y, w, h, confidence, class_id = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
        elif box_format == "xyxy":
            x1, y1, x2, y2, confidence, class_id = box
        else:
            raise ValueError(f"Unsupported box format: {box_format}")

        if confidence > confidence_threshold:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # print(f"Box: {x1}, {y1}, {x2}, {y2}, {confidence}, {class_id}")


def main():
    video_path = "videos/Closeupv2.mp4"
    cap = cv2.VideoCapture(video_path)

    reader = easyocr.Reader(['en'], gpu=False)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Create a MOG2 background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=100, detectShadows=False)

    crop_size = 640
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # out = cv2.VideoWriter('onnx-half.mp4', fourcc,
    #                       25.0, (crop_size, crop_size))

    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

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

                ocr_result = reader.readtext(np.asarray(
                    frame), allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                print(ocr_result)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        # print(f"FPS: {fps}")

        # out.write(frame)
        cv2.imshow("Processed Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
