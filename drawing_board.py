#1.Importing image 
#2.Find hand landmarks
#3.check which fingers are up
#4.If selection mode-Two fingers are up
#5.If drawing mode-Index finger is up 
import cv2 as cv
import time
import numpy as np
import Hand_Tracking_module as htm
import math
import os

folder_path="D:\Python\Advanced computer vision\Canva Designs"
mylist=os.listdir(folder_path)
overlaylist=[]
xp,yp=0,0
imgcanvas=np.zeros((720,1280,3),np.uint8) # we are creating an another blank screen to write 
brushthickness=15

for imgpath in mylist:
    image=cv.imread(f'{folder_path}/{imgpath}')
    overlaylist.append(image)
    canva=overlaylist[0]
    cap=cv.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    detector=htm.HandDetection()
    drawcolor=(255,0,255) #purple default color 

while True:
    #1.importing image
    success,img=cap.read()
    img=cv.flip(img,1)
    #2.finding hand landmarks
    img=detector.findhands(img)
    lmlist=detector.findposition(img,draw=False)
    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]
    #3.check which fingers are up
        fingers=detector.fingersup()
    #print(fingers)
    #4. If selection mode-Two fingers are up
        if fingers[1] and fingers[2]:
           xp,yp=0,0
           #print("Selection Mode")
           if y1<125:            # our canva size is 1280 * 125 based on the selected one we will change the  image 
             if 105<x1<230:    # here 250 and 450 is the region where our 1st colour lies 
                canva=overlaylist[0]
                drawcolor=(255,0,255)
             if 475<x1<665:
                canva=overlaylist[1]
                drawcolor=(255,0,0)
             if 800<x1<870:
                canva=overlaylist[2]
                drawcolor=(0,255,0)
             if 1085<x1<1269:
                canva=overlaylist[3]
                drawcolor=(0,0,0) # black will erase everything 
           cv.rectangle(img,(x1,y1-30),(x2,y2+30),drawcolor,-1)
    
    #5.If drawing mode-Index finger is up
        if fingers[1] and fingers[2]==False:
            cv.circle(img,(x1,y1),15,drawcolor,-1)
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawcolor==(0,0,0): #increasing the size of erasier to rub nicely 
               cv.line(img,(xp,yp),(x1,y1),drawcolor,20)
               cv.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,20)
            else:  
               cv.line(img,(xp,yp),(x1,y1),drawcolor,brushthickness)
               cv.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,brushthickness) # drawing on another blck screen to write 
            xp,yp=x1,y1
            imgGray=cv.cvtColor(imgcanvas,cv.COLOR_BGR2GRAY)
            _,imgInv=cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)
            imgInv=cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
            img=np.bitwise_and(img,imgInv)
            img=np.bitwise_or(img,imgcanvas)
          #print("Drawing Mode")
    #to set canva image
    img[0:125,0:1280]=canva
    img=cv.addWeighted(img,0.5,imgcanvas,0.5,0)
    cv.imshow("Board",img)
    #cv.imshow("Canvas",imgcanvas)
    if cv.waitKey(1) & 0XFF==ord('q'):
        break
