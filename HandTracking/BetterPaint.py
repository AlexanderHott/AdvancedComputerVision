import cv2
import numpy as np
import time
import os
from HandTrackingModule import HandDetector

folderPath = "images/"
imgList = os.listdir(folderPath)
overlay_list = []

for imgPath in imgList:
    image = cv2.imread(folderPath + imgPath)
    overlay_list.append(image)

header = overlay_list[0]
draw_color = (0, 0, 255)
brush_thickness = 40
eraser_thickness = 100
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(maxHands=1, detectionCon=0.8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Hand landmarks

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:

        x1, y1 = lmList[8][1:]  # Index fingertip
        x2, y2 = lmList[12][1:]  # Middle fingertip

        # Find when 1 or 2 fingers up
        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)
            if y1 < 125:
                if 300 < x1 < 500:
                    header = overlay_list[0]
                    draw_color = (0, 0, 255)
                elif 550 < x1 < 750:
                    header = overlay_list[1]
                    draw_color = (0, 255, 0)
                elif 800 < x1 < 950:
                    header = overlay_list[2]
                    draw_color = (0, 255, 255)
                if 1050 < x1 < 1200:
                    header = overlay_list[3]
                    draw_color = (0, 0, 0)

        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
            xp, yp = x1, y1

    img_gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_or(img, imgCanvas)
    img = cv2.bitwise_and(img, img_inv)

    # Set header image
    img[0:125, 0:1280] = header[0:125, 0:1280]
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
