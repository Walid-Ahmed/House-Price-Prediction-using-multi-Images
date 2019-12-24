import os
import cv2


def dataPrep():
	path="HouseImages"

	trainingImages1=[]
	trainingImages2=[]
	trainingImages3=[]
	trainingImages4=[]


	for root, dirs, files in os.walk(path):
	    for name in files:
	    	filePath=os.path.join(root,name)
	    	print(filePath)
	    	img=cv2.imread(filePath)
	    	if ("frontal" in filePath):
	    		trainingImages1.append(img)
	    	if ("kitchen" in filePath):
	    		trainingImages2.append(img)
	    	if ("bathroom" in filePath):
	    		trainingImages3.append(img)
	    	if ("bedroom." in filePath):
	    		trainingImages4.append(img)
	freturn     trainingImages1,trainingImages2,trainingImages3,trainingImages4		
