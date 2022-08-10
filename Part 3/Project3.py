"""
Author : Abhishek Mohabe
Course : CSE 575
"""
import numpy as np
import keras
import seaborn as sns
import scipy.io as io

from matplotlib import pyplot as graphPlot
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Loading the data from the training and test files
trX = io.loadmat('train_32x32.mat')['X']
trY = io.loadmat('train_32x32.mat')['y']
tsX = io.loadmat('test_32x32.mat')['X']
tsY = io.loadmat('test_32x32.mat')['y']

# Images converted into float 64 type
trImages = trX.astype('float64')
tsImages = tsX.astype('float64')

# Labels converted into int64 type
trLabels = trY.astype('int64')
tsLabeles = tsY.astype('int64')

trImages /= 255.0
tsImages /= 255.0

# Transforming labels
labelBn = LabelBinarizer()
trLabels = labelBn.fit_transform(trLabels)
tsLabeles = labelBn.fit_transform(tsLabeles)

# Sequential model for 2D convolution
dataModel = Sequential()
dataModel.add(Conv2D(64, (5, 5), strides = (1, 1), padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
dataModel.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
dataModel.add(Conv2D(64, (5, 5), strides = (1, 1), padding = 'same', activation = 'relu'))
dataModel.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
dataModel.add(Conv2D(128, (5, 5), strides = (1, 1), padding = 'same', activation = 'relu'))

dataModel.add(Flatten())
dataModel.add(Dense(3072, activation = 'relu'))
dataModel.add(Dense(2048, activation = 'relu'))
dataModel.add(Dense(10, activation = 'softmax'))

dataModel.summary()

netOpt = keras.optimizers.SGD(learning_rate = 0.01)
dataModel.compile(optimizer = netOpt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

pred = dataModel.fit(trImages, trLabels, epochs = 20, validation_data = (tsImages, tsLabeles))

trSetAccuracy = pred.history['accuracy']
tsSetAccuracy = pred.history['val_accuracy']
trSetLoss = pred.history['loss']
tsSetLoss = pred.history['val_loss']

graphPlot.figure(figsize = (20, 10))

# Plotting training and testing accuracy
graphPlot.subplot(1, 2, 1)
graphPlot.plot(trSetAccuracy, label ='Training Set Accuracy')
graphPlot.plot(tsSetAccuracy, label ='Testing Set Accuracy')
graphPlot.legend()
graphPlot.title('Epochs vs. Training/Testing Accuracy')

# Plotting training and testing loss
graphPlot.subplot(1, 2, 2)
graphPlot.plot(trSetLoss, label ='Training Set Loss')
graphPlot.plot(tsSetLoss, label ='Testing Set Loss')
graphPlot.legend()
graphPlot.title('Epochs vs. Training/Testing Loss')

graphPlot.show()