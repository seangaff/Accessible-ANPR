import cv2
import numpy as np

videoPath = 'videos/CloseupMini.mp4'
cap = cv2.VideoCapture(videoPath)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
middle_box_width = 400
middle_box_height = 300
middle_box_x1 = int(frame_width/2 - middle_box_width/2)
middle_box_y1 = int(frame_height/2 - middle_box_height/2)
middle_box_x2 = middle_box_x1 + middle_box_width
middle_box_y2 = middle_box_y1 + middle_box_height

# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
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
        cv2.rectangle(frame1, (middle_box_x1, middle_box_y1),
                      (middle_box_x2, middle_box_y2), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement Detected'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
    else:
        cv2.rectangle(frame1, (middle_box_x1, middle_box_y1),
                      (middle_box_x2, middle_box_y2), (0, 0, 255), 2)
        cv2.putText(frame1, "Status: {}".format('No Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    image = cv2.resize(frame1, (1280, 720))
    # out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
# out.release()
