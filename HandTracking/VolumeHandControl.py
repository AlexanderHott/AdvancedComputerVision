import cv2
import time
import numpy as np
from HandTrackingModule import handDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

WCAM, HCAM = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, WCAM)
cap.set(4, HCAM)
pTime = 0

detector = handDetector(maxHands=1, detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)

MINVOL = volRange[0]
MAXVOL = volRange[1]

print("Starting...")
while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 7, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        dist = int(np.sqrt(np.square(x2 - x1) + np.square(y2 - y1)))
        vol = int(np.interp(dist, [10, 150], [MINVOL, MAXVOL]))
        # print(vol)

        volume.SetMasterVolumeLevel(vol, None)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
