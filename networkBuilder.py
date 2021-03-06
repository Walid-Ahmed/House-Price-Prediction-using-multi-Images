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
from keras.layers import concatenate
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import cv2
from keras.utils import plot_model

from util import  dataPrep


def getBaseNetwork(imageInputShape):
	inputA = Input(shape=imageInputShape)
	conv1 = Conv2D(32, kernel_size=4, activation='relu')(inputA)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	flat = Flatten()(pool2)
	net1 = Model(inputs=inputA, outputs=flat)
	net1.summary()
	return net1

def getCommonNetwork(inputs):

	conv1 = Conv2D(32, kernel_size=4, activation='relu')
	pool1 = MaxPooling2D(pool_size=(2, 2))
	conv2 = Conv2D(16, kernel_size=4, activation='relu')
	pool2 = MaxPooling2D(pool_size=(2, 2))
	flat = Flatten()


	output=[]
	for input in inputs:

		x=conv1(input)
		x=pool1(x)
		x=conv2(x)
		x=pool2(x)
		x=flat(x)
		output.append(x)






	return output

def getModel(imageInputShape,weightSharing):

	if (weightSharing):

		#To share a layer across different inputs, simply instantiate the layer once, then call it on as many


		inputA = Input(shape=imageInputShape)
		inputB = Input(shape=imageInputShape)
		inputC = Input(shape=imageInputShape)
		inputD = Input(shape=imageInputShape)

		inputs=[inputA,inputB,inputC,inputD]


		
		output=getCommonNetwork(inputs)





        
		print("Code not ready yet to deal with weight sharing")
		print("Program will now exit")

		combined=concatenate(output)



				# add a dense layer
		dense1=Dense(64,activation='relu')(combined)
		# add a dense layer
		dense2=Dense(64,activation='relu')(dense1)
		# add another dense layer
		out=Dense(1,activation='linear')(dense2)


		model=Model(inputs=[inputA,inputB,inputC,inputD], outputs=out)
		model.summary()

		plot_model(model, to_file='model.png')
		import matplotlib.image as mpimg
		img=mpimg.imread('model.png')
		imgplot = plt.imshow(img)
		plt.show()
		return model


	else:


		branch1=getBaseNetwork(imageInputShape)
		branch2=getBaseNetwork(imageInputShape)
		branch3=getBaseNetwork(imageInputShape)
		branch4=getBaseNetwork(imageInputShape)

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
		return model




