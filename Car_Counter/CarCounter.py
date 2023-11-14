
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

model = YOLO('YOLO_weights/yolov8l.pt')

# For webcam

# cap = cv2.VideoCapture(0)
# cap.set(3,1080)
# cap.set(4,720)

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

mask = cv2.imread("mask.png")

######## Tracking
# max_age - Number of frames to wait before detecting back an object 

tracker = Sort(max_age=20, min_hits=3, iou_threshold = 0.3)

limits = [400,297,673,297]

totalCount = []

# For video 

path = "cars.mp4"
video = cv2.VideoCapture(path)

while True:
    success , img = video.read()
    
    # Overlayimg mask on image

    imgRegion = cv2.bitwise_and(img,mask)

    imgGraphics = cv2.imread('graphics.png', cv2.IMREAD_UNCHANGED)
    img  = cvzone.overlayPNG(img,imgGraphics,(0,0))

    if success:
        # We only pass the region of interest to the detection function
        results =  model(imgRegion,stream = True)

        # Format expected by the tracker (SORT function)
        detections=np.empty((0, 5))

        # Loop through the results and check how it performs for individual bounding boxes
        for result in results:

            ### Bounding box

            boxes = result.boxes

            for box in boxes:

                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w ,h = x2- x1 , y2-y1

                ###### Getting the confidence values

                conf = math.ceil((box.conf[0]*100))/100
                
                ###### Displaying the classes

                cls = box.cls[0]

                #### Selecting the classes of interest
                # Of all the classes (vehicles) being detected, we will need to specify which classes to detect

                currentClass = classNames[int(cls)]

                if currentClass in ['car','bus','truck','motorbike'] and conf > 0.3:
                    # cvzone.cornerRect(img,(x1,y1,w,h), l = 9, rt = 5)
                    # cvzone.putTextRect(img,f'{currentClass} {conf}', (max(20,int(x1)) , max(35,int(y1))),scale = 0.85,thickness = 1, offset =3)

                    # Update detection array
                    # The detections are stacked in the order expected by the tracker
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections,currentArray))
                
                # We notice that when the object of interest is in the centre of frame, the detection is highly accurate
                # Therefore we need to constrain our region of interest (ie choose only a particular region within the frame where the veheicles can be counted)
                
                # Once we obtain the mask, we need to overlay the images and obtain our region of interest
                # Important: The video frame and the maks must have the same shape

                ### Counting the vehicles

                ### Assigning trackers

                # While we have different classes within frame, we need to assign distinct trackers to these objects so that their 
                # movement within the frame can be tracked

                # We will be using SORT( simple online and realtime tracker), which has already ben implemented on github

        # The tracker needs to be updated with a list of detections
        resultsTracker = tracker.update(detections)
        
        # Defining the line beyond which we will count the vehicle
        # Now that we have our IDs, we will count define a line and increase count if the IDs cross that line
        # Coordinates can be obtained via inspection

        cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255), thickness=5)
        
        # Obtain the ID associated with each object 
        for result in resultsTracker:
            x1, y1, x2, y2 , ID = result
            print(result)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w ,h = x2- x1 , y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h), l = 9, rt = 2,colorR=(255,0,0))
            cvzone.putTextRect(img,f'{int(ID)}', (max(20,int(x1)) , max(35,int(y1))),scale = 2,thickness = 1, offset = 10)
        
            # Finding centre of the bounding boxes
            cx, cy = x1 + w//2 , y1 + h//2
            cv2.circle(img,(cx,cy),5,(0,255,0),cv2.FILLED)

            ##### Updating count
            # If the region is too broad, there are multiple detections for same object but in case of a narrow region there might not be detections at all
            # To counter this, we store the IDs in a list and avoid duplicate entries
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if ID not in totalCount:
                    totalCount.append(ID)
                    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0), thickness=5)

        #cvzone.putTextRect(img,f'Count: {len(totalCount)}', (50,50))
        cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

        cv2.imshow('Image',img)
        #cv2.imshow('Image Region',imgRegion)
        cv2.waitKey(1)

cv2.destroyAllWindows()
#cap.release()

