# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:38:37 2017

@author: Karan Vijay Singh
"""
## Convolution Neural Network
import keras
import numpy as np

from keras.datasets import mnist
from PIL import Image
from keras import backend as back
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import matplotlib.pyplot as plt


##Loading Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_create = []
test_create = []

## Number of Training & test Points
trainingPts = 6000
testPts = 1000

#taking one tenth of test and training pts
y_test = y_test[0:testPts]
y_train = y_train[0:trainingPts]

#Resizing images as array 32,32 for training pts
for i in range(trainingPts):
    #picking the image from train set
	xImage = Image.fromarray(x_train[i])
    #resizing the image
	n_Image = xImage.resize((32,32), Image.HAMMING)
	imageArr = n_Image.convert('L')
	imageArr = np.array(imageArr)
    #appending the image
	train_create.append(imageArr)
    
#Resizing images as array 32,32 for test pts    
for i in range(testPts):
    #picking the image from test set
	xImage = Image.fromarray(x_test[i])
    #resizing the image
	n_testImage = xImage.resize((32,32), Image.HAMMING)
	imageArr = n_testImage.convert('L')
	imageArr = np.array(imageArr)
    #appending the image
	test_create.append(imageArr)

#creating an array of training images
x_train = np.array(train_create)
#creating an array of testing images
x_test = np.array(test_create)

values=255
row = 32
column = 32
# adding extra dim to train and test set for CNN compatibility
if back.image_data_format() == 'channels_first':
    #reshaping the training images
	x_train = x_train.reshape(x_train.shape[0], 1, row, column)
    #reshaping the testing images
	x_test = x_test.reshape(x_test.shape[0], 1, row, column)
	dim = (1, row, column)
else:
    #reshaping the training images
	x_train = x_train.reshape(x_train.shape[0], row, column, 1)
    #reshaping the testing images
	x_test = x_test.reshape(x_test.shape[0], row, column, 1)
	dim = (row, column, 1)

no_of_classes = 10    
#conversion to binary matrixes
y_train = keras.utils.to_categorical(y_train, no_of_classes)
y_test = keras.utils.to_categorical(y_test, no_of_classes)

## Changing Data type as float and normalising the data
x_train = x_train.astype('float32')
x_train /= values
x_test = x_test.astype('float32')
x_test /= values

####################################################################
#Initialising CNNModel
CNNModel = Sequential()
#Adding layers to the CNNModel

#Adding Convolutional layer 1
CNNModel.add(Conv2D(64, (3,3), input_shape = dim, activation='relu',padding='same'))
#Adding MaxPooling layer 2
CNNModel.add(MaxPooling2D(pool_size=(2,2)))
#Adding Convolutional layer 3
CNNModel.add(Conv2D(128, (3,3), activation='relu',padding='same'))
#Adding MaxPooling layer 4
CNNModel.add(MaxPooling2D(pool_size=(2,2))) 
#Adding Convolutional layer 5
CNNModel.add(Conv2D(256, (3,3), activation='relu',padding='same'))

#Adding Convolutional layer 6
#CNNModel.add(Conv2D(256, (3,3), activation='relu',padding='same'))
#Adding MaxPooling layer 7
#CNNModel.add(MaxPooling2D(pool_size=(2,2))) 
#Adding Convolutional layer 8
#CNNModel.add(Conv2D(512, (3,3), activation='relu',padding='same'))
#Adding Convolutional layer 9
#CNNModel.add(Conv2D(512, (3,3), activation='relu',padding='same'))
#Adding MaxPooling layer 10
#CNNModel.add(MaxPooling2D(pool_size=(2,2)))
#Adding Convolutional layer 11
#CNNModel.add(Conv2D(512, (3,3), activation='relu',padding='same'))
#Adding Convolutional layer 12
#CNNModel.add(Conv2D(512, (3,3), activation='relu',padding='same'))

#Adding MaxPooling layer 13
CNNModel.add(MaxPooling2D(pool_size=(2,2)))
#Flatten
CNNModel.add(Flatten())
#Adding Flattening layer 14
CNNModel.add(Dense(units = 4096, activation = 'relu'))
#Adding Flattening layer 15
CNNModel.add(Dense(units = 4096, activation = 'relu'))
#Adding Flattening layer 16
CNNModel.add(Dense(units = 512, activation = 'relu'))
#Adding Flattening layer 17
CNNModel.add(Dense(units = 10, activation = 'softmax'))
# Compiling CNN using adam as optimiser
CNNModel.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
#############################################################################################
#print(CNNModel.output_shape)


# Fitting the CNNmodel 
result = CNNModel.fit(x_train, y_train, batch_size=32, validation_data=(x_test,y_test), epochs=3, verbose=1)
#calculating scores
score = CNNModel.evaluate(x_test, y_test, batch_size=32)

# Plotting accuracy using summarised history for accuracy
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy (Train and Test) vs Epochs')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.grid()
plt.show()

# Plotting loss using summarised history for accuracy
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss (Train and Test) vs Epochs')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.grid()
plt.show()