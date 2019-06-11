# import the necessary packages
%matplotlib inline
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10] #width * height

image = cv2.imread('Scanned.jpg')


gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(7,7),1.41)
edge=auto_canny_otsu(blur)
plt.imshow(edge,"gray")
plt.title("Gaussian blur")
plt.show()


equ = cv2.equalizeHist(gray)
plt.imshow(equ,"gray")
plt.title("Equalize Histogram")
plt.show()

plt.imshow(gray,"gray")
plt.title("Normal Histogram")
plt.show()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
plt.imshow(cl1,"gray")
plt.title("CLAHE")
plt.show()


plt.imshow(edge,"gray")
plt.title("Normal Canny edge")
plt.show()

canny_equa=auto_canny_otsu(equ)
canny_equa=cv2.bilateralFilter(canny_equa,3,50,50)
plt.imshow(canny_equa,"gray")
plt.title("Canny edge Equalize")
plt.show()

canny_clahe=auto_canny_otsu(cl1)
canny_clahe=cv2.bilateralFilter(canny_clahe,3,50,50)
plt.imshow(canny_clahe,"gray")
plt.title("CLAHE")
plt.show()


_, contours, _ = cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)



mincontour=500
maxcontour=20000
filteredContours=[]

for c in contours :
    area=cv2.contourArea(c)
    peri=cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,peri*0.002,True)
    if peri>mincontour and peri<maxcontour: 
        filteredContours.append(approx)


cnts=cv2.drawContours(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB), filteredContours, -1, (0,255,0), 3)
plt.imshow(cnts,'gray')
print("Number of filtered contours: ",len(filteredContours))