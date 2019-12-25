import os
import cv2
import pandas as pd
from keras.preprocessing.image import img_to_array
import numpy as np



def dataPrep(width,height,normalizeFlag):

	priceDict=dict()
	print("[INFO] loading house attributes...")
	inputPath =  "HousesInfo.txt"
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
	prices=df["price"].tolist()
	#prices = [ int(x) for x in prices ]
	maximumPrice=max(prices)

	if(normalizeFlag):
		for i in range(len(prices)):
			prices[i]=prices[i]/maximumPrice

	print(prices)	
		

	houseID=1
	for price in prices:
		priceDict[houseID]=float(price)
		print("[INFO] Price of HOUSE with id {} is {}".format(houseID,price))
		houseID=houseID+1
	print("-----------------------------------------------------------------------------------")
	

	print("[INFO] Loading images..............")






	path="HouseImages"

	trainingImages1=[]
	trainingImages2=[]
	trainingImages3=[]
	trainingImages4=[]
	trainY=[]


	houseIds=[]  #growing list of house ids added to trainY
	for root, dirs, files in os.walk(path):
	    for name in files:
	    	if('.DS' in  name):
	    		continue
	    	filePath=os.path.join(root,name)
	    	houseID=int(name.split("_")[0])
	    	#print(houseID,filePath)

	    	if not (houseID in houseIds):
	    		trainY.append(priceDict[houseID])
	    		houseIds.append(houseID)
	    		print("[INFO] Adding images of house with id {} and price {}".format(houseID,priceDict[houseID]))



	    		

	    	img=cv2.imread(filePath)
	    	img = cv2.resize(img, (width,height))

	    	img = img_to_array(img)	

	    	if ("frontal" in filePath):
	    		trainingImages1.append(img)
	    	if ("kitchen" in filePath):
	    		trainingImages2.append(img)
	    	if ("bathroom" in filePath):
	    		trainingImages3.append(img)
	    	if ("bedroom." in filePath):
	    		trainingImages4.append(img)


    

	x1 = np.array(trainingImages1, dtype="float") / 255.0
	x2 = np.array(trainingImages2, dtype="float") / 255.0
	x3 = np.array(trainingImages3, dtype="float") / 255.0
	x4 = np.array(trainingImages4, dtype="float") / 255.0

	for houseID,price in zip(houseIds,trainY): 
		print("[INFO] House with id {} and price {}".format(houseID,price))

	print("-----------------------------------------------------------------------------------")


	return     x1,x2,x3,x4, np.array(trainY),maximumPrice		