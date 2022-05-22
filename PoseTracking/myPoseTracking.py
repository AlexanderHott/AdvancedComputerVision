import cv2
import mediapipe as mp
import time

from PoseTrackingModule import poseDetector

cap = cv2.VideoCapture("videos/video-1.mp4")
pTime = 0
detector = poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img, draw=True)
    lmList = detector.findPosition(img, draw=True)
    if lmList:
        cv2.circle(img, (lmList[0][1], lmList[0][2]), 15, (0, 255, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(10)
