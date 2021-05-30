import cv2
import mediapipe as mp
import time

from HandTrackingModule import handDetector

pTime = 0
ctime = 0
cap = cv2.VideoCapture(0)  # Webcam ID
detector = handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)