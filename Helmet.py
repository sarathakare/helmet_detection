'''
helmet detection
'''
import math
from ultralytics import YOLO
import cv2
import cvzone


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r'C:\Users\yk_th\Downloads\helmet_detection_v1\helmet_detection_v1\motorbikes_-_62 (360p) - Trim.mp4')
cap.set(3,1280)
cap.set(4,720)
 
model = YOLO(r'C:\Users\yk_th\OneDrive\Documents\helmet\best.pt')

classNames = ['With Helmet', 'Without Helmet']


while True:
    success,img = cap.read()
    results = model(img, stream = True)

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int (x1),int (y1),int (x2),int (y2)
            w = x2-x1
            h = y2-y1
            currentClass = classNames[int(box.cls[0])]
            #if currentClass == "bike"  or currentClass == "bicycle" or currentClass == "motorbike":
            cvzone.cornerRect(img,(x1,y1,w,h), l = 20)
            conf = math.floor(box.conf[0]*100)/100                
            cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),
                                scale=0.6,thickness=1, offset=5)







    cv2.imshow("Image",img) 
    cv2.waitKey(1)

