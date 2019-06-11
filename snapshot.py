import cv2
import datetime
import os 
import time 
import imutils 


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
	
	cv2.imshow("Image", imutils.resize(frame,height=650))
	
	k = cv2.waitKey(1)
	if k == 27:
		cap.release()
		break
	elif k==32:
		for x in range(1,2):
			
			if newdir==False:
				os.mkdir(dir)
				os.chdir(dir)
				newdir=True
		
			# FileName='Snapshot '+str(i)+'.png'
			# cv2.imwrite(FileName, frame)

			FileName2='Snapshot '+str(i)+'.jpg'
			cv2.imwrite(FileName2, frame)

			print("Image "+str(i)+" written")
			i+=1
			time.sleep(0.1)
			
	
cv2.destroyAllWindows()


