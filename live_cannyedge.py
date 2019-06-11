# Importing libraries

import cv2
import numpy as np
import datetime
from matplotlib import pyplot as plt
import os

# Creating new directory and file name extension
now = datetime.datetime.now()
date = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"-" + \
	str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)
dir = "NTUA CannyEdge " + date

# Defining a function that does nothing. It is required p


def nothing(x):
	pass


cap = cv2.VideoCapture(0)
"""
The number in VideoCapture is either 0,1 or 2.
It's the index of the chosen camera
The index can change depending on which device is connected on boot
If the external webcam is connected on boot, the number is 0
IF the external webcam is connected after booting, the number is 1 or 2
Depending on which usb port it is connected.
"""


"""
Define the function that set the definition of the camera
By default it is set to 480p
The first argument of cap.set() is for chosing whether we want to change the number of pixel
horizontally (3) or vertically (4)
"""


def set720p():
	cap.set(3, 1280)
	cap.set(4, 720)
	return


def set480p():
	cap.set(3, 640)
	cap.set(4, 480)
	return


def set1080p():
	cap.set(3, 1920)
	cap.set(4, 1080)
	return


# Counter used for naming files
i = 1


# set1080p()
set720p()
# set480p()

# Creating and naming a windows for paramaters
cv2.namedWindow('Parameters')

# create trackbars for parameters changes
# A function is needed, we use the function "nothing" defined earlier
# The arguments are : name of the trackbar, name of the window, start value, end value, function
cv2.createTrackbar('MinThreshold', 'Parameters', 0, 255, nothing)
cv2.createTrackbar('MaxThreshold', 'Parameters', 100, 255, nothing)
cv2.createTrackbar('Gaussian Kernel', 'Parameters', 7, 15, nothing)
cv2.createTrackbar('Median Filtering', 'Parameters', 0, 15, nothing)
cv2.createTrackbar('Bilateral Filtering', 'Parameters', 0, 1, nothing)


# Boolean used for checking if a new directory has been created
newdir = False


# while loop used during the acquisition
while(1):
	ret, frame = cap.read()
	# ret return a boolean. True if we can read from the camera
	# frame is the object returned from the reading of the camera

	# If a flip is needed, use the code below
	# frame=cv2.flip(frame,1)

	# Creating and updating the values linked to the trackbars
	# Arguments are : name of the trackbar, name of the window
	minthreashold = cv2.getTrackbarPos('MinThreshold', 'Parameters')
	maxthreashold = cv2.getTrackbarPos('MaxThreshold', 'Parameters')
	gaussiankernel = cv2.getTrackbarPos('Gaussian Kernel', 'Parameters')
	medfiltkernel = cv2.getTrackbarPos('Median Filtering', 'Parameters')
	bilat = cv2.getTrackbarPos('Bilateral Filtering', 'Parameters')

	# Converting to gray scale, we don't need the information on colors
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Gaussian filtering and median filtering need the kernel value to be a odd number
	if gaussiankernel % 2 == 0 and gaussiankernel != 0:
		gaussiankernel = gaussiankernel+1
		#print("Gaussian kernel not an odd number, reassigned value is %d" %gaussiankernel)

	if medfiltkernel % 2 == 0 and medfiltkernel != 0:
		medfiltkernel = medfiltkernel+1
		#print("Median filter kernel not an odd number, reassigned value is %d" %medfiltkernel)

		# Apply Gaussian or median filtering only if the number of the kernel is greater than zero
	if gaussiankernel != 0:
		frame = cv2.GaussianBlur(frame, (gaussiankernel, gaussiankernel), 1.41)
	# else:
		#print("Gaussian filtering is inactive")

	if medfiltkernel != 0:
		frame = cv2.medianBlur(frame, medfiltkernel)
	# else:
		#print("Median filtering is inactive")

	# Toggle the Bilateral filtering
	if bilat == 1:
		frame = cv2.bilateralFilter(frame, 9, 75, 75)
		print("Bilateral filtering is active")
	# else:
		#print("Bilateral filtering is inactive")

		# Apply a canny edge detector with selected threashold
	edge = cv2.Canny(frame, minthreashold, maxthreashold)

	cv2.imshow("Parameters", edge)
	cv2.imshow("Image", frame)

	k = cv2.waitKey(1)
	if k == 27:  # escape button
		cap.release()
		break
	elif k == 32:  # space button
		# Condition to create only one directory for each run of the application
		if newdir == False:
			os.mkdir(dir)
			os.chdir(dir)
			newdir = True

		# Naming of the file and saving the image to a png file
		CannyEdgeFile = 'Canny edge'+str(i)+'.png'
		cv2.imwrite(CannyEdgeFile, edge)

		# Converting the image to an numpy array and binarize it
		img_np = np.array(edge)
		img_np = img_np > 100

		# Saving the array to a csv file
		np.savetxt("CannyEdge "+str(i)+".csv",
				   img_np, delimiter=",", fmt='% 4d')

		plt.matshow(img_np)
		plt.imshow(img_np, cmap="Greys")
		plt.savefig("CSV plot "+str(i))

		# Saving the source image used for the processing
		SourceImageFile = 'Source image'+str(i)+'.png'
		cv2.imwrite(SourceImageFile, frame)

		print("Image "+str(i)+" written")
		i += 1


cv2.destroyAllWindows()
