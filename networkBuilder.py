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



inputA = Input(shape=(64,64,1))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(inputA)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
net1 = Model(inputs=inputA, outputs=flat)
net1.summary()


inputA = Input(shape=(64,64,1))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(inputA)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
net2= Model(inputs=inputA, outputs=flat)
net2.summary()


inputA = Input(shape=(64,64,1))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(inputA)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
net3 = Model(inputs=inputA, outputs=flat)
net3.summary()


inputA = Input(shape=(64,64,1))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(inputA)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
net4 = Model(inputs=inputA, outputs=flat)
net4.summary()



combined=concatenate([net1.output,net2.output,net3.output,net4.output])

dense1=Dense(64,activation='relu')(combined)
dense2=Dense(64,activation='relu')(dense1)
out=Dense(1,activation='linear')(dense2)


model=Model(inputs=[net1.input,net2.input,net3.input,net4.input], outputs=out)
model.summary()

plot_model(model, to_file='model.png')
import matplotlib.image as mpimg
img=mpimg.imread('model.png')
imgplot = plt.imshow(img)
plt.show()

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

model.fit([trainingImages1,trainingImages2,trainingImages3,trainingImages4],trainY,validation_data=([testImages1,testImages2,testImages3,testImages4],testY),epochs=200,batch_size=8)



