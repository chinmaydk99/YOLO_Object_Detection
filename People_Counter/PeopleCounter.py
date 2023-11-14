from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

model = YOLO('YOLO_Weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('mask.png')

tracker = Sort(max_age=20, min_hits=3, iou_threshold= 0.3)

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

cap = cv2.VideoCapture('people.mp4')

totalCountUp = []
totalCountDown = []

while True:
    success, frame = cap.read()
    frameRegion = cv2.bitwise_and(frame,mask)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, imgGraphics, (730, 260))

    if success:
        results = model(frameRegion, stream = True)

        detections = np.empty((0,5))

        for result in results:

            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1, y2-y1

                # Confidence values

                conf = math.ceil(box.conf[0]* 100) / 100
                
                # Displaying the classes

                cls = box.cls[0]

                currentClass = classNames[int(cls)]

                if currentClass == 'person' and conf > 0.3:
                    cvzone.cornerRect(frame,(x1,y1,w,h), l = 9, rt = 5)
                    cvzone.putTextRect(frame, f'{currentClass}:{conf}', (max(20,x1),max(35,y1)), scale = 0.85, thickness = 1, offset = 3)

                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack([detections,currentArray])

        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, ID = result
            x1, y1, x2, y2, ID = int(x1), int(y1), int(x2), int(y2), int(ID)
            w,h = x2-x1 , y2-y1

            cvzone.cornerRect(frame,(x1,y1,w,h), l = 9, rt = 2,colorR=(255,0,0))
            cvzone.putTextRect(frame,f'{int(ID)}', (max(20,int(x1)) , max(35,int(y1))),scale = 2,thickness = 1, offset = 10)

            # Finding the center of bounding boxes
            cx,cy = x1 + w//2 , y1 + h//2
            cv2.circle(frame,(cx,cy),5,(0,255,0),cv2.FILLED)

            cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
            cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

            #print(cx,cy)

            # Updating count
            if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-15 < cy < limitsUp[1]+15:
                if ID not in totalCountUp:
                    totalCountUp.append(ID)
                    cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
            
            if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-15 < cy < limitsDown[1]+15:
                if ID not in totalCountDown:
                    totalCountDown.append(ID)
                    cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
                
        cv2.putText(frame,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
        cv2.putText(frame,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)

        cv2.imshow('Video',frame)
        #cv2.imshow('ROI',frameRegion)

        if cv2.waitKey(1) & 0xFF  == ord('q'):
            break