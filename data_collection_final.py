import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback

# Initialize the webcam capture and hand detectors
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Count existing images in directory A and set the current directory to 'A'
base_dir = r"D:\new_project\American-sign-Language-main\Final Project\Source Code\AtoZ_3.1"
count = len(oss.listdir(oss.path.join(base_dir, 'A')))
c_dir = 'A'

# Parameters for image processing
offset = 15
step = 1
flag = False
suv = 0

# Create a white image for background
white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("./white.jpg", white)

while True:
    try:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for Mediapipe
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = hd.findHands(imgRGB, draw=False, flipType=True)
        white = cv2.imread("./white.jpg")

        # Debugging: Check the content of hands
        print("hands type:", type(hands))
        print("hands content:", hands)

        if hands:
            hand = hands[0]  # Accessing the first hand detected
            # Debugging: Check the content of hand
            print("hand type:", type(hand))
            print("hand content:", hand)

            # Ensure 'bbox' exists and is within the frame dimensions
            if 'bbox' in hand:
                x, y, w, h = hand['bbox']

                # Ensure coordinates are within the frame dimensions
                x1, y1 = max(0, x - offset), max(0, y - offset)
                x2, y2 = min(frame.shape[1], x + w + offset), min(frame.shape[0], y + h + offset)
                image = np.array(frame[y1:y2, x1:x2])

                handz, imz = hd2.findHands(image, draw=True, flipType=True)
                if handz:
                    hand = handz[0]  # Accessing the first hand in the cropped image
                    pts = hand['lmList']
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15

                    # Draw lines between keypoints on the hand
                    for t in range(0, 4, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

                    skeleton0 = np.array(white)
                    zz = np.array(white)
                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                    skeleton1 = np.array(white)

                    cv2.imshow("1", skeleton1)

        # Add text to the frame and display it
        frame = cv2.putText(frame, f"dir={c_dir}  count={count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)

        # Handle keyboard interrupts
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:  # ESC key to exit
            break

        if interrupt & 0xFF == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) == ord('Z') + 1:
                c_dir = 'A'
            flag = False
            count = len(oss.listdir(oss.path.join(base_dir, c_dir)))

        if interrupt & 0xFF == ord('a'):
            flag = not flag
            suv = 0 if flag else suv

        print("=====", flag)
        if flag:
            if suv == 180:
                flag = False
            if step % 3 == 0:
                save_dir = oss.path.join(base_dir, c_dir)
                save_path = oss.path.join(save_dir, f"{count}.jpg")
                cv2.imwrite(save_path, skeleton1)

                count += 1
                suv += 1
            step += 1

    except Exception:
        print("==", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
