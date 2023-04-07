import cv2
import numpy as np
import onnx
import onnxruntime as ort


def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
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
    # Load the ONNX model
    onnx_model_path = 'models/bestv2-half.onnx'
    # onnx_model = onnx.load(onnx_model_path)
    # onnx.checker.check_model(onnx_model)

    # Create an ONNX runtime session
    ort_session = ort.InferenceSession(onnx_model_path)
    input_dims = ort_session.get_inputs()[0].shape
    target_size = (input_dims[3], input_dims[2])
    print(f"Model input dimensions: {input_dims}")

    # Preprocess the input image
    input_image_path = 'car.jpeg'
    preprocessed_image = preprocess_image(input_image_path, target_size)

    # Run the ONNX model
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    boxes = ort_session.run([output_name], {input_name: preprocessed_image})[0]
    print(f"Boxes shape: {boxes.shape}, Boxes content: {boxes}")

    # Post-process the output and draw bounding boxes
    original_image = cv2.imread(input_image_path)
    # resize original image to match the input dimensions of the model
    original_image = cv2.resize(original_image, (640, 640))
    box_format = "xywh"  # Change this to "xyxy" if your model outputs boxes in that format
    draw_bounding_boxes(original_image, boxes[0], box_format)
    # for box in boxes[0]:
    #     if box_format == "xywh":
    #         x, y, w, h, confidence, class_id = box
    #         x1, y1 = int(x - w / 2), int(y - h / 2)
    #         x2, y2 = int(x + w / 2), int(y + h / 2)
    #     elif box_format == "xyxy":
    #         x1, y1, x2, y2, confidence, class_id = box
    #     else:
    #         raise ValueError(f"Unsupported box format: {box_format}")

    #     if confidence > 0.5:
    #         cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         print(f"Box: {x1}, {y1}, {x2}, {y2}, {confidence}, {class_id}")

    # Display the image with bounding boxes
    cv2.imshow('Output', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image
    output_image_path = 'output_image.png'
    cv2.imwrite(output_image_path, original_image)


if __name__ == '__main__':
    main()
