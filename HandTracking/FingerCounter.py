import cv2
import time
from HandTrackingModule import handDetector

WCAM, HCAM = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, WCAM)
cap.set(4, HCAM)
pTime = 0

detector = handDetector(maxHands=1, detectionCon=0.75)

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    fingers = [0, 0, 0, 0, 0]

    if lmList:
        y0, y1, y2 = lmList[17][2], lmList[4][2], lmList[3][2]  # pinky base, thumb fingertip, thumb lower
        if abs(y1 - y0) > abs(y2 - y0):
            fingers[0] = 1

        y3, y4 = lmList[8][2], lmList[6][2]  # Index fingertip, index lower
        if y4 > y3:
            fingers[1] = 1

        y5, y6 = lmList[12][2], lmList[10][2]  # middle fingertip, middle lower
        if y6 > y5:
            fingers[2] = 1

        y7, y8 = lmList[16][2], lmList[14][2]  # ring fingertip, ring lower
        if y8 > y7:
            fingers[3] = 1

        y9, y10 = lmList[20][2], lmList[18][2]  # pinky fingertip, pinky lower
        if y10 > y9:
            fingers[4] = 1

    cv2.putText(img, f"Fingers: {fingers}", (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
