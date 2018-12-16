#https://machinelearningmastery.com/keras-functional-api-deep-learning/


#The model has 10 inputs, 3 hidden layers with 10, 20, and 10 neurons, 
#and an output layer with 1 output. Rectified linear activation functions 
#are used in each hidden layer and a sigmoid activation function is used 
#in the output layer, for binary classification

#demo of keras functional API
#advantages over sequential API
#It specifically allows you to define multiple input or output models 
#as well as models that share layers.
# More than that, it allows you to define ad hoc acyclic network graphs.
#Multi,ayer Perceptron
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

#The input layer takes a shape argument that is a tuple that indicates the
#dimensionality of the input data.

#When input data is one-dimensional, such as for a multilayer Perceptron, 
#the shape must explicitly leave room for the shape of the mini-batch size used
# when splitting the data when training the network. Therefore, the shape tuple 
#is always defined with a hanging last dimension when the input is one-dimensional (2,)
visible = Input(shape=(10,)) # 10 inputs
hidden1 = Dense(10, activation = 'relu')(visible) # hidden layer1 with 10 neurons,  dense layer connects the input layer output as the input to the dense hidden layer1
hidden2 = Dense(20, activation = 'relu')(hidden1)
hidden3 = Dense(10, activation = 'relu')(hidden2)
outputLayer = Dense(1,activation='sigmoid')(hidden3) #2 is the num of classes here
model = Model(inputs=visible, outputs=outputLayer)

# summarize layers
print(model.summary())
# plot graph
from keras.utils import plot_model
plot_model(model, to_file='multilayer_perceptron_graph.png')


#The model receives black and white 64×64 images as input, 
#then has a sequence of two convolutional and pooling layers as feature extractors,
# followed by a fully connected layer to interpret the features
# and an output layer with a sigmoid activation for two-class predictions.

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
CNN = Input(shape = (64,64,1))
conv1 = Conv2D(32, kernel_size = 4, activation = 'relu')(CNN)
maxPool1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv2 = Conv2D(32, kernel_size = 4, activation = 'relu')(maxPool1)
maxPool2 = MaxPooling2D(pool_size=(2,2))(conv2)
hiddenCNN = Dense(10, activation='relu')(maxPool2)
output = Dense(1,activation='sigmoid')(hiddenCNN)
CNNModel = Model(inputs = CNN, outputs = output)
# summarize layers
print(CNNModel.summary())
# plot graph
plot_model(CNNModel, to_file='convolutional_neural_network.png')


#The model expects 100 time steps of one feature as input.
# The model has a single LSTM hidden layer to extract features from the sequence, 
# followed by a fully connected layer to interpret the LSTM output, 
# followed by an output layer for making binary predictions.

from keras.layers.recurrent import LSTM

visibleRNN = Input(shape=(100,1))
hiddenRNN1 = LSTM(10)(visibleRNN) # 10 memory cells
hiddenRNN2 = Dense(10, activation='relu')(hiddenRNN1)
outputRNN = Dense(1, activation='sigmoid')(hiddenRNN2)
modelRNN = Model(inputs=visibleRNN, outputs=outputRNN)
# summarize layers
print(modelRNN.summary())
# plot graph
plot_model(modelRNN, to_file='recurrent_neural_network.png')


#n this section, we define multiple convolutional layers with differently sized 
#kernels to interpret an image input.

#The model takes black and white images with the size 64×64 pixels. 
#There are two CNN feature extraction submodels that share this input; 
#the first has a kernel size of 4 and the second a kernel size of 8. 
#The outputs from these feature extraction submodels are flattened into vectors 
#and concatenated into one long vector and passed on to a fully connected layer for
# interpretation before a final output layer makes a binary classification.

from keras.layers import Flatten
from keras.layers.merge import concatenate
CNN1 = Input(shape = (64,64,1))
# first feature extractor
convCNN1 = Conv2D(32, kernel_size = 4, activation = 'relu')(CNN1)
maxPoolCNN1 = MaxPooling2D(pool_size=(2,2))(convCNN1)
flattenCNN1 = Flatten()(maxPoolCNN1)
# second feature extractor
convCNN2 = Conv2D(32, kernel_size=8, activation= 'relu')(CNN1) # kernel size is basically filter size, sometimes it is called kernel width
maxPoolCNN2 = MaxPooling2D(pool_size=(2,2), activation = 'relu')(convCNN2)
flattenCNN2 = Flatten()(maxPoolCNN2)
# merge feature extractors
mergeCNN = concatenate([flattenCNN1,flattenCNN2])
# interpretation layer
hiddenCNN1 = Dense(10, activation= 'relu')(mergeCNN)
#prediction output
outputCNN12 = Dense(1, activation='sigmoid')(hiddenCNN1)

modelCNN12 = Model(inputs = CNN1, outputs = outputCNN12)
# summarize layers
print(modelCNN12.summary())
# plot graph
plot_model(modelCNN12, to_file='shared_input_layer.png')

#Shared Feature Extraction Layer
#The input to the model is 100 time steps of 1 feature. 
#An LSTM layer with 10 memory cells interprets this sequence.
# The first interpretation model is a shallow single fully connected layer, 
#the second is a deep 3 layer model. The output of both interpretation models 
#are concatenated into one long vector that is passed to the output layer
# used to make a binary prediction
# define input
visible = Input(shape=(100,1))
# feature extraction
extract1 = LSTM(10)(visible)
# first interpretation model
interp1 = Dense(10, activation='relu')(extract1)
# second interpretation model
interp11 = Dense(10, activation='relu')(extract1)
interp12 = Dense(20, activation='relu')(interp11)
interp13 = Dense(10, activation='relu')(interp12)
# merge interpretation
merge = concatenate([interp1, interp13])
# output
output = Dense(1, activation='sigmoid')(merge)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='shared_feature_extractor.png')

#Multiple Input Model
#We will develop an image classification model that takes two versions of the image
# as input, each of a different size. Specifically a black and white 64×64 version 
#and a color 32×32 version. Separate feature extraction CNN models operate on each,
# then the results from both models are concatenated for interpretation and 
#ultimate prediction.

# first input model
visible1 = Input(shape=(64,64,1))
conv11 = Conv2D(32, kernel_size=4, activation='relu')(visible1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(16, kernel_size=4, activation='relu')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
flat1 = Flatten()(pool12)
# second input model
visible2 = Input(shape=(32,32,3))
conv21 = Conv2D(32, kernel_size=4, activation='relu')(visible2)
pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
conv22 = Conv2D(16, kernel_size=4, activation='relu')(pool21)
pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
flat2 = Flatten()(pool22)
# merge input models
merge = concatenate([flat1, flat2])
# interpretation model
hidden1 = Dense(10, activation='relu')(merge)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
#Note that in the creation of the Model() instance, 
#that we define the two input layers as an array. Specifically:
model = Model(inputs=[visible1, visible2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multiple_inputs.png')



