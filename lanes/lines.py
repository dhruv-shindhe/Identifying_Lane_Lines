import cv2
import numpy as np
from matplotlib import pyplot as plt

def finalPtsOfLRlines(img,line):
    slope, intercept = line
    y1=int(img.shape[0])
    y2=int(3/5*y1)
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return [x1,y1,x2,y2]



def avgSlopeIntercept(img,lines):
    leftLane=[]
    rightLane=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            parameters=np.polyfit((x1,x2),(y1,y2),1)
            slope=parameters[0]
            intercept=parameters[1]
            if slope<0:
                leftLane.append((slope,intercept))
            else:
                rightLane.append((slope,intercept))
    leftAvg=np.average(leftLane,axis=0)
    rightAvg=np.average(rightLane,axis=0)
    #the above give the slope and intercept of l and r lane next func covert it to 2 differen  pts so that it could b eplotted
    leftPt=finalPtsOfLRlines(img,leftAvg)
    rightPt=finalPtsOfLRlines(img,rightAvg)
    return [leftPt,rightPt]

def roi(img):
    heigth=img.shape[0]
    triangle=np.array([[(200,heigth),(1100,heigth),(550,250)]])
    mask=np.zeros_like(img)
    masked=cv2.bitwise_and(cv2.fillPoly(mask,triangle,255),img)
    return masked

def displayLines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1=line[0]
            y1=line[1]
            x2=line[2]
            y2=line[3]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image



def main():
    # img=cv2.imread("laneImg2.jpg")
    # laneImg=np.copy(img)
    # grey=cv2.cvtColor(laneImg,cv2.COLOR_RGB2GRAY)
    # blur=cv2.GaussianBlur(grey,(5,5),0)
    # canImg=cv2.Canny(blur,50,150)
    # houghLines=cv2.HoughLinesP(roi(canImg),2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    # singleLine=avgSlopeIntercept(laneImg,houghLines)
    # laneline=displayLines(laneImg,singleLine)
    # combinedImg=cv2.addWeighted(laneImg,0.8,laneline,1,1)
    # cv2.imshow('lanes',combinedImg)
    # cv2.waitKey(0)

    cap=cv2.VideoCapture("test2.mp4")
    while(cap.isOpened()):
        _,frame=cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        canImg = cv2.Canny(blur, 50, 150)
        houghLines = cv2.HoughLinesP(roi(canImg), 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        singleLine = avgSlopeIntercept(frame, houghLines)
        laneline = displayLines(frame, singleLine)
        combinedImg = cv2.addWeighted(frame, 0.8, laneline, 1, 1)
        cv2.imshow('lanes', combinedImg)
        if cv2.waitKey(1)==ord('q'):
            break

main()

