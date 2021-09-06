import cv2
import numpy as np


address='http://192.168.29.41:8080/video'
cap=cv2.VideoCapture(address)

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)

    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):
    # cv2.RETR_EXTERNAL get the extrme cornes(mode)
    biggest=np.array([])
    maxArea=0
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)  # (contour,random resolution ,shape is  closed or not)
            if area>maxArea and len(approx)==4:
                biggest =approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

def Preprocessing(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlurr=cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlurr,100,200)

    #sometimes edges are too thin, we can use dialation function to make them thicker and then Erosion fun to make it thinner again
    kernel =np.ones((5,5),np.uint8)
    imgDial=cv2.dilate(imgCanny,kernel,iterations=2)
    imgThreshold=cv2.erode(imgDial,kernel,iterations=1)
    return imgThreshold

def reorder(myPoints):
    #reshape the biggest
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),np.int32)
    add=myPoints.sum(1) #axis 1 we add row wise each element
    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[2] =myPoints[np.argmax(add)]
    diff=np.diff(myPoints,axis=1)
    myPointsNew[3]=myPoints[np.argmin(diff)]
    myPointsNew[1] = myPoints[np.argmax(diff)]
    return  myPointsNew

def getWarsp(img,biggest):
    biggest=reorder(biggest)
    print(biggest)
    pt1 = np.float32(biggest)
    width=480
    height=640
    pt2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # transform matrix
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))

    #clear noises around the boundary
    imgCropped =imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped=cv2.resize(imgCropped,(width,height))
    return  imgCropped

while True:
    success,img=cap.read()
    img=cv2.resize(img,(480,640))
    imgContour=img.copy()

    imgThr=Preprocessing(img)
    biggest= getContours(imgThr)
    if biggest.size !=0:
        imgWarp=getWarsp(img,biggest)
        imgArray=[img,imgContour,imgThr,imgWarp]
    else :
        imgArray = [img, img, imgThr, img]

    imgStack=stackImages(0.6,imgArray)
    cv2.imshow("image",imgStack)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()