"""
Author : Abhishek Mohabe
Course : CSE 575
"""

import scipy.io
import scipy.stats
import numpy as np
import statistics

# Loading the fileData
fileData = scipy.io.loadmat('mnist_data.mat')

# filedata classified into training and testing
trainingFiledata = fileData['trX']
trainingLabel = fileData['trY']
testingFiledata = fileData['tsX']
testingLabel = fileData['tsY']

# Average and Standard deviation for training data

trainingAvg = []
trainingStddev = []
trainingFiledataLen = len(trainingFiledata[0])

for i in range(len(trainingFiledata)):
    trainingAvg.append(sum(trainingFiledata[i])/trainingFiledataLen)
    trainingStddev.append(statistics.stdev(trainingFiledata[i]))

# Average and Standard deviation for testing data

testingAvg = []
testingStddev = []
testingFiledataLen = len(testingFiledata[0])

for i in range(len(testingFiledata)):
    testingAvg.append(sum(testingFiledata[i])/testingFiledataLen)
    testingStddev.append(statistics.stdev(testingFiledata[i]))

# X and Y training fileData

ndList = [trainingAvg, trainingStddev]

X = np.asarray(ndList)
y = trainingLabel

testndList = [testingAvg, testingStddev]
Xtest = np.asarray(testndList)

# Prediction Method

def predict(X):
    linearModel = np.dot(X, weights) + bias
    yPredicted = sigmoid(linearModel)
    
    yPredicted_class = [1 if i > 0.5 else 0 for i in yPredicted]
    return yPredicted_class

# Accuracy calculation

def accuracy(yTrue, yPred):
    accuracy = np.sum(yTrue == yPred) / len(yTrue)
    return accuracy

# Sigmoid value calculation

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Gradient ascent model 

def gradientModel(X, y):
    nSamples = len(trainingAvg)
    epochIters = 1000
    weights = np.zeros(3)
    bias = 0
    learningRate = 0.1

    for _ in range(epochIters):
        linearModel = np.dot(X, weights) + bias
        yPredicted = sigmoid(linearModel)
        dw = (1 / nSamples) * np.dot(X.T, (yPredicted - y))
        db = (1 / nSamples) * np.sum(yPredicted - y)

        weights -= learningRate * dw
        bias -= learningRate * db

gradientModel(X, y)
predictions = predict(Xtest)

print("Logistic Regression accuracy:", accuracy(yTest, predictions))