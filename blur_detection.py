# import the necessary packages
from imutils import paths
import argparse
import cv2
import glob
 
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True,
# 	help="path to input directory of images")
# ap.add_argument("-t", "--threshold", type=float, default=100.0,
# 	help="focus measures that fall below this value will be considered 'blurry'")
# args = vars(ap.parse_args())


images = glob.glob('*.jpg')
threshold=100

# loop over the input images
for imagePath in images:
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)


	if fm > threshold:
		text = imagePath+" - Not Blurry: "+str(fm)
		print(imagePath+" - Not Blurry: "+str(fm))
 
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if fm <threshold:
		text = imagePath+" - Blurry: "+str(fm)
		print(imagePath+" - Blurry: "+str(fm))