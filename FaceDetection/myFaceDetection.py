import cv2
import mediapipe as mp
import time

from FaceDetectionModule import FaceDetector

# cap = cv2.VideoCapture("videos/video-1.mp4")
cap = cv2.VideoCapture(0)
pTime = 0
detector = FaceDetector()
while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=True)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2
    )

    cv2.imshow("Image", img)
    cv2.waitKey(30)
