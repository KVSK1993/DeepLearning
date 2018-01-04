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
test_create_rotate = []
degreeValues = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
scoresValues = []
degree_rotate = ['-45', '-40', '-35', '-30', '-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40', '45']
## Number of Training & test Points
trainingPts = 6000
testPts = 1000

#taking one tenth of test and training pts
y_train = y_train[0:trainingPts]
y_test = y_test[0:testPts]

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
'''for i in range(testPts):
	xImage = Image.fromarray(x_test[i])
	n_testImage = xImage.resize((32,32), Image.HAMMING)
	imageArr = n_testImage.convert('L')
	imageArr = np.array(imageArr)
	test_create.append(imageArr)'''

#creating an array of training images
x_train = np.array(train_create)
#x_test = np.array(test_create)

no_of_classes = 10
#conversion to binary matrixes
y_train = keras.utils.to_categorical(y_train, no_of_classes)
y_test = keras.utils.to_categorical(y_test, no_of_classes)


row = 32
column = 32
# adding extra dim to train and test set for CNN compatibility
if back.image_data_format() == 'channels_first':
    #reshaping the training images
	x_train = x_train.reshape(x_train.shape[0], 1, row, column)
    #reshaping the testing images
	#x_test = x_test.reshape(x_test.shape[0], 1, row, column)
	dim = (1, row, column)
else:
    #reshaping the training images
	x_train = x_train.reshape(x_train.shape[0], row, column, 1)
    #reshaping the testing images
	#x_test = x_test.reshape(x_test.shape[0], row, column, 1)
	dim = (row, column, 1)
    
values=255
## Changing Data type as float and normalising the data
x_train = x_train.astype('float32')
x_train /= values
#x_test = x_test.astype('float32')
#x_test /= 255

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
CNNModel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#############################################################################################
#print(CNNModel.output_shape)


# Fitting the CNNmodel 
result = CNNModel.fit(x_train, y_train, batch_size=32, epochs=3, verbose=1)

#score = CNNModel.evaluate(x_test, y_test, batch_size=32)


## Looping over all the entire test set to rotate all images by specific degrees

for d in degreeValues:
    test_create_rotate = []
    for i in range(testPts):
        #picking the image from test set 
        testImage = Image.fromarray(x_test[i])
        #resizing the image as required
        nImage = testImage.resize((32,32), Image.HAMMING)
        # Rotating the image as required
        rImage = nImage.rotate(d)
        imageArrayRot = rImage.convert('L')
        #appending the image
        test_create_rotate.append(np.array(imageArrayRot))
        #creating an array of rotated images
    testCreateRotated = np.array(test_create_rotate)
    #reshaping the rotated image
    if back.image_data_format() == 'channels_first':
        testCreateRotated = testCreateRotated.reshape(testCreateRotated.shape[0], 1, row, column)
    else:
        testCreateRotated = testCreateRotated.reshape(testCreateRotated.shape[0], row, column,1)
    ## Changing Data type as float and normalising the data
    testCreateRotated = testCreateRotated.astype('float32')
    testCreateRotated /= values
    #evaluating the model after rotation
    score = CNNModel.evaluate(testCreateRotated, y_test, batch_size=32)
    scoresValues.append(score)

scoresValues = np.array(scoresValues)
        

# Plotting accuracy using summarised history for accuracy
plt.plot(degreeValues, scoresValues[:,1])
plt.xticks(degreeValues, degree_rotate)
plt.title('TestSet: Accuracy vs Rotation')
plt.ylabel('Accuracy on test')
plt.xlabel('Degrees of rotation')
plt.grid()
plt.show()

# Plotting loss using summarised history for accuracy
plt.plot(degreeValues, scoresValues[:,0])
plt.xticks(degreeValues, degree_rotate)
plt.title('TestSet: Loss vs Rotation')
plt.ylabel('Loss on test')
plt.xlabel('Degrees of rotation')
plt.grid()
plt.show()
