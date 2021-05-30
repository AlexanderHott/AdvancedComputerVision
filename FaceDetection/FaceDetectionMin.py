import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/video-1.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for _id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            # cv2.rectangle(img, bbox.xmin, bbox.height, (255, 0, 255), 2, )
            cv2.rectangle(img, bbox, (255, 0, 255), 2, )
            cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime=cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(30)
