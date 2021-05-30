import cv2
from HandTrackingModule import handDetector
import numpy as np

WCAM, HCAM = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, WCAM)
cap.set(4, HCAM)

detector = handDetector(maxHands=1, detectionCon=0.5)


while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        x1, y1 = lmList[4][1:]
        x2, y2 = lmList[8][1:]

        dist = int(np.sqrt(np.square(x2 - x1) + np.square(y2 - y1)))
        thickness = min(max(int(200/max(dist, 1)), 1), 10)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), thickness)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
