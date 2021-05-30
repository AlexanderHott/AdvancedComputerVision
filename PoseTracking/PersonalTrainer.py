import math

import cv2
import numpy as np
from PoseTrackingModule import PoseDetector


cap = cv2.VideoCapture("videos/video-2.mp4")
detector = PoseDetector()
count = 0
goingUp = True


while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # detector.findAngle(img, 11, 13, 15)     # Left arm
        angle = detector.findAngle(img, 12, 14, 16)     # Right arm

        percent = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (210, 310), (650, 110))
        # Check for curls
        if percent == 100:
            if goingUp:
                count += 0.5
                goingUp = False
        if percent == 0:
            if not goingUp:
                count += 0.5
                goingUp = True

        cv2.rectangle(img, (1100, 100), (1175, 650), (0, 255, 0), 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{int(percent)}%", (1100, 650), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

        cv2.putText(img, f"curls: {math.floor(count)}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
