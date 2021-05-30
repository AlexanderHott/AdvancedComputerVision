import cv2
import mediapipe as mp

import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNum=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]

            for _id, lm in enumerate(myHand.landmark):
                # print(_id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(_id, cx, cy)
                self.lmList.append((_id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 5), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = [0, 0, 0, 0, 0]
        y0, y1, y2 = self.lmList[17][2], self.lmList[4][2], self.lmList[3][
            2]  # pinky base, thumb fingertip, thumb lower
        if abs(y1 - y0) > abs(y2 - y0):
            fingers[0] = 1

        y3, y4 = self.lmList[8][2], self.lmList[6][2]  # Index fingertip, index lower
        if y4 > y3:
            fingers[1] = 1

        y5, y6 = self.lmList[12][2], self.lmList[10][2]  # middle fingertip, middle lower
        if y6 > y5:
            fingers[2] = 1

        y7, y8 = self.lmList[16][2], self.lmList[14][2]  # ring fingertip, ring lower
        if y8 > y7:
            fingers[3] = 1

        y9, y10 = self.lmList[20][2], self.lmList[18][2]  # pinky fingertip, pinky lower
        if y10 > y9:
            fingers[4] = 1

        return fingers


def main():
    pTime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)  # Webcam ID
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
