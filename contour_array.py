# import the necessary packages
%matplotlib inline
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10] #width * height
import pandas as pd


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# test_image="test_size_6.jpg"
test_image="Scanned1.jpg"
# test_image="Snapshot 1.jpg"


# width=25.75 #2 € 
width=18.75 #2 centime
#  A4 210 × 297 


# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(test_image)
# image=imutils.resize(image, height=650)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
# plt.imshow(gray,'gray')

height, width2 = image.shape[:2]
pixelsPerMetric2=width2/297

#testing thresholding
Block_size=11
constant=3
# th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,Block_size,constant)
_,th = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# _,cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(gray,cnts , -1, (255,0,0), 1)
# cv2.imshow("contours",gray)
plt.imshow(th,"gray")
plt.title('Thresolding')
plt.show()

out = np.zeros_like(th)
# _,cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
_,cnts,_ = cv2.findContours(th, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

cv2.drawContours(gray,cnts , -1, (255,0,0), 1)
plt.imshow(gray,'gray')
plt.title("Contour")
plt.show()

# find contours in the edge map

#simple canny edge
# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#auto canny edge
# cnts = cv2.findContours(auto_edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#thresolding
th2=cv2.bitwise_not(th)
_,cnts,_ = cv2.findContours(th2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)


pixelsPerMetric = None
orig = image.copy()

orig=cv2.cvtColor(orig,cv2.COLOR_BGR2RGB)
orig2=orig.copy()
contour_list=[]


    
# keeping only large contour
for c in cnts:
    if cv2.contourArea(c) > 100:
        contour_list.append(c)
        
# finding the maximum contour in the contour
max_cnt=0 
for cnt in contour_list:
    if len(cnt) >max_cnt:
        max_cnt=len(cnt)



numero_contour=0
element_contour=0
ligne=0 #always 0
x=0 # always 0
y=1 # always 1


# print(contour_list[numero_contour][element_contour][ligne][x])
# print(contour_list[numero_contour][element_contour][ligne][y])
dictio={}
df=pd.DataFrame(dictio)
df1=pd.DataFrame(dictio)
for i in range(len(contour_list)):
    list_x=[]
    list_y=[]
    for j in range(len(contour_list[i])):
        list_x.append(contour_list[i][j][ligne][x])
        list_y.append(contour_list[i][j][ligne][y])
    
    title_x="x"+str(i)
    title_y="y"+str(i)
    df1[title_x]=pd.Series(list_x)
    df1[title_y]=pd.Series(list_y)
    df=pd.concat([df,df1],axis=1,sort=False)
    df1=pd.DataFrame(dictio)

df.head()
# df.shape
# df

# saving the dataframe of contour's coordinate into a csv file
df.to_csv("coordinates.csv")