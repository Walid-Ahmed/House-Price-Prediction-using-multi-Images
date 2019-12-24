import os
import cv2
import pandas as pd


def dataPrep():


	priceDict=dict()
	print("[INFO] loading house attributes...")
	inputPath =  "HousesInfo.txt"
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
	prices=df["price"].tolist()

	houseID=1
	for price in prices:
		priceDict[houseID]=int(price)
		houseID=houseID+1
		print(houseID)

	print(len(prices))	

	print(priceDict[135])



	path="HouseImages"

	trainingImages1=[]
	trainingImages2=[]
	trainingImages3=[]
	trainingImages4=[]
	trainY=[]



	for root, dirs, files in os.walk(path):
	    for name in files:
	    	if('.DS' in  name):
	    		continue
	    	filePath=os.path.join(root,name)
	    	houseID=int(name.split("_")[0])
	    	print(houseID,filePath)
	    	trainY.append(priceDict[houseID])
	    	img=cv2.imread(filePath)
	    	if ("frontal" in filePath):
	    		trainingImages1.append(img)
	    	if ("kitchen" in filePath):
	    		trainingImages2.append(img)
	    	if ("bathroom" in filePath):
	    		trainingImages3.append(img)
	    	if ("bedroom." in filePath):
	    		trainingImages4.append(img)




	


	return     trainingImages1,trainingImages2,trainingImages3,trainingImages4,trainY		