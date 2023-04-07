import cv2
import onnxruntime as ort
import numpy as np
import tensorflow as tf
# from PIL import Image
import cv2


def preprocess_image(image, target_size):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(
        image_normalized, (2, 0, 1))  # Change to NCHW format
    image_expanded = np.expand_dims(image_transposed, axis=0)
    return image_expanded.astype(np.float16)


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
            print(f"Box: {x1}, {y1}, {x2}, {y2}, {confidence}, {class_id}")


def main():
    video_path = "videos/Closeupv2.mp4"
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Load the ONNX model
    onnx_model_path = 'models/bestv2-half.onnx'

    # Create an ONNX runtime session
    ort_session = ort.InferenceSession(onnx_model_path)
    input_dims = ort_session.get_inputs()[0].shape
    target_size = (input_dims[3], input_dims[2])
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print(f"Model input dimensions: {input_dims}")

    # Create a MOG2 background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=100, detectShadows=False)

    crop_size = 640
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # out = cv2.VideoWriter('filtered_video.mp4', fourcc,
    #                       25.0, (crop_size, crop_size))

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

                # Run the ONNX model
                preprocessed_image = preprocess_image(
                    frame, target_size)

                boxes = ort_session.run(
                    [output_name], {input_name: preprocessed_image})[0]
                print(f"Boxes shape: {boxes.shape}, Boxes content: {boxes}")

                box_format = "xywh"  # Change this to "xyxy" if your model outputs boxes in that format
                draw_bounding_boxes(frame, boxes[0], box_format)

        # out.write(frame)
        cv2.imshow("Processed Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
