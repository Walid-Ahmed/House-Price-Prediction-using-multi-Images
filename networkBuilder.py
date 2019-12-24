#import the necessary packages


# Convolutional Neural Network
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.utils import plot_model
import matplotlib.pyplot as plt
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
	return     trainingImages1,trainingImages2,trainingImages3,trainingImages4		



def getBaseNetwork():
	inputA = Input(shape=(64,64,1))
	conv1 = Conv2D(32, kernel_size=4, activation='relu')(inputA)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	flat = Flatten()(pool2)
	net1 = Model(inputs=inputA, outputs=flat)
	net1.summary()
	return net1



branch1=getBaseNetwork()
branch2=getBaseNetwork()
branch3=getBaseNetwork()
branch4=getBaseNetwork()

combined=concatenate([branch1.output,branch2.output,branch3.output,branch4.output])

# add a dense layer
dense1=Dense(64,activation='relu')(combined)
# add a dense layer
dense2=Dense(64,activation='relu')(dense1)
# add another dense layer
out=Dense(1,activation='linear')(dense2)


model=Model(inputs=[branch1.input,branch2.input,branch3.input,branch4.input], outputs=out)
model.summary()

plot_model(model, to_file='model.png')
import matplotlib.image as mpimg
img=mpimg.imread('model.png')
imgplot = plt.imshow(img)
plt.show()

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)


trainingImages1,trainingImages2,trainingImages3,trainingImages4	=dataPrep()
model.fit([trainingImages1,trainingImages2,trainingImages3,trainingImages4],trainY,validation_data=([testImages1,testImages2,testImages3,testImages4],testY),epochs=200,batch_size=8)



