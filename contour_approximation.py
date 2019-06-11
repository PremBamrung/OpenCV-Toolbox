# import the necessary packages
%matplotlib inline
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10] #width * height

image = cv2.imread('Scanned.jpg')
# image=cv2.bitwise_not(image)

def auto_canny_otsu(img):
    otsu_thresh_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    canny = cv2.Canny(img, otsu_thresh_val, otsu_thresh_val*0.5)
    return canny


gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gaussian_kernel=11
# blur=cv2.GaussianBlur(gray,(gaussian_kernel,gaussian_kernel),1.41)
blur = cv2.bilateralFilter(gray, 5, 175, 175)
edge=auto_canny_otsu(blur)
# edge=cv2.Canny(blur,100,200)
plt.imshow(edge,"gray")
plt.title("Gaussian blur")
plt.show()



_, contours, _ = cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)


    
mincontour=200
maxcontour=20000
filteredContours=[]

for c in contours :
    area=cv2.contourArea(c)
    peri=cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,peri*0.002,True)
    if peri>mincontour and peri<maxcontour: 
        filteredContours.append(c)


# cnts=cv2.drawContours(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB), filteredContours, -1, (0,255,0), 3)
cnts=cv2.drawContours(np.zeros_like(image), filteredContours, -1, (255,255,255), 3)
plt.imshow(cnts,'gray')
print("Number of filtered contours: ",len(filteredContours))


print(image.dtype)
print(cnts.dtype)

gray= cv2.cvtColor(cnts, cv2.COLOR_RGB2BGR)
# gray=cv2.bitwise_not(gray)
plt.imshow(gray)
gray.max()

# image = cv2.imread('test_size_6.jpg',0)
# image=imutils.resize(image,width=650)
# image = cv2.medianBlur(image,5)
output=image.copy()

try:
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,60,param1=50,param2=60,minRadius=20,maxRadius=1000)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
    #     cv2.imshow("output", np.hstack([image, output]))
        plt.imshow(output,"gray")
    else:
        print("No circles were found")

except:
    print("HoughCircles didn't work")
    
    
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
# plt.imshow(cimg)

# detect circles in the image
# circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
 
