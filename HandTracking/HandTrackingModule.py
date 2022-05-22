import cv2
import mediapipe as mp
import time
import typing as t


class HandDetector:
    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 2,
        detectionCon: float = 0.5,
        trackCon: float = 0.5,
    ):
        self.mode = mode

        self.maxHands = maxHands

        self.detectionCon = detectionCon

        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.detectionCon,
            self.trackCon,
        )

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True) -> t.NamedTuple:

        self.results = self.hands.process(img)

        if self.results.multi_hand_landmarks:

            for handLms in self.results.multi_hand_landmarks:

                if draw:

                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        return img

    def findPosition(self, img, handNum=0, draw=True):

        self.lmList = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNum]

            for _id, lm in enumerate(myHand.landmark):

                h, w, c = img.shape

                cx, cy = int(lm.x * w), int(lm.y * h)

                self.lmList.append((_id, cx, cy))

                if draw:

                    cv2.circle(img, (cx, cy), 7, (255, 0, 5), cv2.FILLED)

        return self.lmList

    def fingersUp(self):

        fingers = [0, 0, 0, 0, 0]

        # pinky base, thumb fingertip, thumb lower

        y0, y1, y2 = (
            self.lmList[17][2],
            self.lmList[4][2],
            self.lmList[3][2],
        )

        if abs(y1 - y0) > abs(y2 - y0):

            fingers[0] = 1

        # Index fingertip, index lower

        y3, y4 = self.lmList[8][2], self.lmList[6][2]

        if y4 > y3:

            fingers[1] = 1

        # middle fingertip, middle lower

        y5, y6 = (
            self.lmList[12][2],
            self.lmList[10][2],
        )

        if y6 > y5:

            fingers[2] = 1

        # ring fingertip, ring lower

        y7, y8 = self.lmList[16][2], self.lmList[14][2]

        if y8 > y7:

            fingers[3] = 1

        # pinky fingertip, pinky lower

        y9, y10 = self.lmList[20][2], self.lmList[18][2]

        if y10 > y9:

            fingers[4] = 1

        return fingers


def main():

    pTime = 0

    cTime = 0

    cap = cv2.VideoCapture(0)  # Webcam ID

    detector = HandDetector()

    while True:

        success, img = cap.read()

        img = detector.findHands(img)

        lmList = detector.findPosition(img)

        if lmList:

            print(lmList[4])

        cTime = time.time()

        fps = 1 / (cTime - pTime)

        pTime = cTime

        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
