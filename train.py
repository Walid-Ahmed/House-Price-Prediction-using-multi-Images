#python train.py --numOfEpochs  20 --width  128 --height 128 --channels 3  --normalizeFlag True --weightSharing False  --batchSize  32
#python train.py --numOfEpochs  20 --width  128 --height 128 --channels 3  --normalizeFlag True --weightSharing False   --batchSize  32


from  networkBuilder import getModel
from util import  dataPrep
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import argparse


def train(numOfEpochs):
	imageInputShape=(width,height,channels)
	model=getModel(imageInputShape,weightSharing)
	opt = Adam(lr=1e-3, decay=1e-3 / 200)
	model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
	trainingImages1,trainingImages2,trainingImages3,trainingImages4,prices,maximumPrice	=dataPrep(width,height,normalizeFlag)
	(trainX1, testX1, trainX2, testX2, trainX3, testX3, trainX4, testX4, trainY, testY) = train_test_split(trainingImages1,trainingImages2,trainingImages3,trainingImages4,prices, test_size=0.25, random_state=42)


	print("[INFO] Dataset of trainX1 shape {}".format(trainX1.shape))
	print("[INFO] Dataset of testX1 shape {}".format(testX1.shape))
	print("[INFO] Dataset of trainX2 shape {}".format(trainX2.shape))
	print("[INFO] Dataset of testX2 shape {}".format(testX2.shape))
	print("[INFO] Dataset of trainX3 shape {}".format(trainX3.shape))
	print("[INFO] Dataset of testX3 shape {}".format(testX3.shape))
	print("[INFO] Dataset of trainX4 shape {}".format(trainX4.shape))
	print("[INFO] Dataset of testX4 shape {}".format(testX4.shape))

	print("[INFO] Dataset of trainY shape {}".format(trainY.shape))
	print("[INFO] Dataset of testY shape {}".format(testY.shape))

	print("-----------------------------------------------------------------------------------")



	history=model.fit([trainX1,trainX2,trainX3,trainX4],trainY,validation_data=([testX1,testX2,testX3,testX4],testY),epochs=numOfEpochs,batch_size=batchSize)
	validationLoss=(history.history['val_loss'])
	trainingLoss=history.history['loss']




	#------------------------------------------------
	# Plot training and validation accuracy per epoch
	epochs   = range(len(validationLoss)) # Get number of epochs
	 #------------------------------------------------
	plt.plot  ( epochs,     trainingLoss ,label="Training Loss")
	plt.plot  ( epochs, validationLoss, label="Validation Loss" )
	plt.title ('Training and validation loss')
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	fileToSaveAccuracyCurve="plot_acc.png"
	plt.savefig("plot_acc.png")
	print("[INFO] Loss curve saved to {}".format("plot_acc.png"))
	plt.legend(loc="upper right")
	plt.show()

	return model,maximumPrice,testX1,testX2,testX3,testX4,testY



if __name__ == "__main__":



	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("--numOfEpochs", required=True, help="path to image file")
	ap.add_argument("--width", required=True, help="path to image file")
	ap.add_argument("--height", required=True, help="path to image file")
	ap.add_argument("--channels", default=3, help="path to image file")
	ap.add_argument("--normalizeFlag", default=False, help="path to image file")
	ap.add_argument("--weightSharing", default=False, help="path to image file")
	ap.add_argument("--batchSize", default=32, help="path to image file")






	#read the arguments
	args = vars(ap.parse_args())
	numOfEpochs=int(args["numOfEpochs"])
	width=int(args["width"])
	height=int(args["height"])
	channels=int(args["channels"])
	normalizeFlag=args["normalizeFlag"]
	weightSharing=args["weightSharing"]
	batchSize=int(args["batchSize"])

	print(normalizeFlag)

	if (normalizeFlag=="True"):
		normalizeFlag=True
	else:
		normalizeFlag=False	

	if (weightSharing=="True"):
		weightSharing=True
	else:
		weightSharing=False	


	model,maximumPrice,testX1,testX2,testX3,testX4,testY=train(numOfEpochs)



	# make predictions on the testing data
	print("[INFO] predicting house prices...")
	preds = model.predict([testX1,testX2,testX3,testX4])
	print(preds)


	if(normalizeFlag):
		#readjust house prices
		testY=testY*maximumPrice
		preds=preds*maximumPrice



	plt.plot  ( testY ,label="Actual price")
	plt.plot  ( preds, label="Predicted price" )
	plt.title ('House prices')
	plt.xlabel("Point #")
	plt.ylabel("Price")
	plt.legend(loc="upper right")
	plt.savefig("HousePrices.png")
	plt.show()
	print("[INFO] predicted vs actual price saved to HousePrices.png")
