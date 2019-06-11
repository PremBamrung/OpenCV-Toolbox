import cv2
import datetime
import os 
import time 
import imutils 
import numpy as np


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def nothing(x):
	pass

now= datetime.datetime.now()
date= str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"-"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)
dir="Snapshot "+ date

cap = cv2.VideoCapture(2)

def set720p():
	cap.set(3, 1280)
	cap.set(4, 720)
	return 

def set480p():
	cap.set(3, 640)
	cap.set(4, 480)

def set1080p():
	cap.set(3, 1920)
	cap.set(4, 1080)	

i=1
newdir=False

set1080p()


while(1):
	ret, frame = cap.read()
	
	ratio = frame.shape[0] / 500.0
	orig = frame.copy()
	frame = imutils.resize(frame, height = 500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 0, 100)
	cv2.imshow("Edged", edged)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	 
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break





	cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Outline", frame)



	k = cv2.waitKey(1)
	if k == 27:
		cap.release()
		break
	elif k==32:
	
			
		if newdir==False:
			os.mkdir(dir)
			os.chdir(dir)
			newdir=True
	
		# FileName='Snapshot '+str(i)+'.png'
		# cv2.imwrite(FileName, frame)
		warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)


		FileName2='Snapshot '+str(i)+'.jpg'
		cv2.imwrite(FileName2, warped)

		print("Image "+str(i)+" written")
		cv2.imshow("Image "+str(i),warped)
		i+=1


			
	
cv2.destroyAllWindows()


