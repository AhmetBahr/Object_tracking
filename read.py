import cv2 
import numpy as np
from obje_Class import ObjectDetection
from tracker import *

#Initialize Object Detection

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture('Video/highway.mp4') 
#cap = cv2.VideoCapture('Video/los_angeles.mp4')

od = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height , width, _ = frame.shape

    
    roi = frame[340: 1000, 500: 1200]


    mask = od.apply(roi) # mask = od.apply(frame)
    _, mask = cv2.threshold(mask, 254,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
           # cv2.drawContours(frame,[cnt],-1,(0,255,0),2)
           x,y,w,h = cv2.boundingRect(cnt)
           cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)

           detections.append([x,y,w,h])

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv2.putText(roi,str(id),(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)


    cv2.imshow("roi" , roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()

    
 

