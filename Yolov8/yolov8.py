import cv2
import math
import cvzone
from ultralytics import YOLO

# capture the webcam
#cap = cv2.VideoCapture(0)
#cap.set(3,3280)
#cap.set(4,2000)

#capture the video

cap = cv2.VideoCapture('T:/Advance AI/cnn/videos/pexels_videos_2034115 (1080p).mp4')

# model of yolov8 weights
model=YOLO('yolov8n.pt')

# class file
class_name=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard',
'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
'scissors', 'teddy bear', 'hair drier', 'toothbrush']

while True:
    res,frame = cap.read()
    results=model(frame,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding boxes
            x1 ,y1 ,x2 ,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)

            w,h=x2-x1,y2-y1
            cvzone.cornerRect(frame,(x1,y1,w,h),l=9)
            # confidence score
            conf =math.ceil((box.conf[0] *100))/100
            # class
            cls = int(box.cls[0])
            cvzone.putTextRect(frame,f'{class_name[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.5,thickness=1,offset=5)



    cv2.imshow('image',frame)
    cv2.waitKey(1)
