"""
Author : Abhishek Mohabe
Course : CSE 575
"""

import scipy.io
import scipy.stats
import statistics

#Loading filedata from mat file
filedata = scipy.io.loadmat('mnist_data.mat')

#filedata classified into training and testing
trainingFiledata = filedata['trX']
trainingLabel = filedata['trY']
testingFiledata = filedata['tsX']
testingLabel = filedata['tsY']

# Average and Standard Deviation calculation for training data

trainingAvg = []
trainingStddev = []
trainingFiledataLen = len(trainingFiledata[0])

for i in range(len(trainingFiledata)):
    trainingAvg.append(sum(trainingFiledata[i])/trainingFiledataLen)
    trainingStddev.append(statistics.stdev(trainingFiledata[i]))

# Average and Standard Deviation calculation for testing data

testingAvg = []
testingStddev = []
testingFiledataLen = len(testingFiledata[0])

for i in range(len(testingFiledata)):
    testingAvg.append(sum(testingFiledata[i])/testingFiledataLen)
    testingStddev.append(statistics.stdev(testingFiledata[i]))

trainingAvgArray = {}
trainingStddevArray = {}
testingAvgArray = {}
testingStddevArray = {}


for i in range(len(trainingAvg)):
    classLabel = int(trainingLabel[0][i])
    if classLabel not in trainingAvgArray:
        trainingAvgArray[classLabel] = list()
    trainingAvgArray[classLabel].append(trainingAvg[i])
    

for i in range(len(trainingAvg)):
    classLabel = int(trainingLabel[0][i])
    if classLabel not in trainingStddevArray:
        trainingStddevArray[classLabel] = list()
    trainingStddevArray[classLabel].append(trainingStddev[i])
    
    
for i in range(len(testingAvg)):
    classLabel = int(testingLabel[0][i])
    if classLabel not in testingAvgArray:
        testingAvgArray[classLabel] = list()
    testingAvgArray[classLabel].append(testingAvg[i])
    

for i in range(len(testingAvg)):
    classLabel = int(testingLabel[0][i])
    if classLabel not in testingStddevArray:
        testingStddevArray[classLabel] = list()
    testingStddevArray[classLabel].append(testingStddev[i])   

trainingAvg0 = trainingAvgArray[0]
trainingAvg1 = trainingAvgArray[1]
trainingStddev0 = trainingStddevArray[0]
trainingStddev1 = trainingStddevArray[1]

testingAvg0 = testingAvgArray[0]
testingAvg1 = testingAvgArray[1]
testingStddev0 = testingStddevArray[0]
testingStddev1 = testingStddevArray[1]

# Mean and Standard deviation for each variables x1 and x2 for both classes 0 and 1:

avg0X1 = sum(trainingAvg0)/len(trainingAvg0)
std0X1 = statistics.stdev(trainingAvg0)
avg1X1 = sum(trainingAvg1)/len(trainingAvg1)
std1X1 = statistics.stdev(trainingAvg1)
avg0X2 = sum(trainingStddev0)/len(trainingStddev0)
std0X2 = statistics.stdev(trainingStddev0)
avg1X2 = sum(trainingStddev1)/len(trainingStddev1)
std1X2 = statistics.stdev(trainingStddev1)

print("Values for X1")
print("Average-class 0: ", avg0X1)
print("Average-class 1: ", avg1X1)
print("Standard Deviation-class 0: ", std0X1)
print("Standard Deviation-class 1: ", std1X1)
print("Values for X2")
print("Average-class 0: ", avg0X2)
print("Average-class 1: ", avg1X2)
print("Standard Deviation-class 0: ", std0X2)
print("Standard Deviation-class 1: ", std1X2)

# Gaussian Distribution

gaussian0X1 = scipy.stats.norm(avg0X1, std0X1)
gaussian1X1 = scipy.stats.norm(avg1X1, std1X1)
gaussian0X2 = scipy.stats.norm(avg0X2, std0X2)
gaussian1X2 = scipy.stats.norm(avg1X2, std1X2)

# Naive Bayes calculation using Gaussian distribution

predicted0 = []
predicted1 = []

for i in range(len(testingAvg0)):
    prob0Avg0= gaussian0X1.pdf(testingAvg0[i]) * gaussian0X2.pdf(testingStddev0[i]) * 0.5
    prob1Avg0 = gaussian1X1.pdf(testingAvg0[i]) * gaussian1X2.pdf(testingStddev0[i]) * 0.5
    
    if(prob0Avg0 > prob1Avg0):
        predicted0.append(0)
    else:
        predicted0.append(1)

for i in range(len(testingAvg1)):
    prob0Avg1 = gaussian0X1.pdf(testingAvg1[i]) * gaussian0X2.pdf(testingStddev1[i]) * 0.5
    prob1Avg1 = gaussian1X1.pdf(testingAvg1[i]) * gaussian1X2.pdf(testingStddev1[i]) * 0.5
    
    if(prob0Avg1 < prob1Avg1):
        predicted1.append(1)
    else:
        predicted1.append(0)

# Calculating the Accuracy

count0 = 0
count1 = 0
for i in range(len(predicted0)):
    if predicted0[i] == 0:
        count0 += 1
    if predicted0[i] == 1:
        count1 += 1

print("Naive Bayes Algorithm accuracy: ", ((count0+count1)/len(testingFiledata)) * 100, "%")