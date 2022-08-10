"""
Author : Abhishek Mohabe
Course : CSE 575
"""
import scipy.io, numpy as np,random, sys,math, matplotlib.pyplot as s1graphPlot
from scipy.spatial import distance

dataFile = scipy.io.loadmat("AllSamples.mat")['AllSamples']
dataFileDimensions = dataFile.shape

print("K-Means Strategy 1" + "\n")
nCenters = []
x = 'c'
initial = 0
centers = []
cCentroid = {}
cPoints = {}

while initial <= 1:
	J = []
	for i in range(2,11):
		centers = []
		cCentroid = {}
		cPoints = {}
		nCenters = []
		#generating random indices
		randIndices = np.random.choice(dataFileDimensions[0], i, replace = False)
		#assigning random points as centers to the list
		for j in randIndices:
			centers.append(dataFile[j])

		for i in range(len(randIndices)):
			nCenters.append(None)

		count = 1
		nCenters = np.array(nCenters)
		#storing cluster centroid points
		for center in centers:
			cCentroid["Cluster" + str(count)] = center
			count = count + 1

		centers = np.array(centers)
		intermList = []
		#assigning centroid points to cluster
		for key,val in cCentroid.items():
			intermList.append(list(val))
			cPoints[key] = intermList
			intermList = []
		#assigning data points until centroid value remains same	
		flag = False
		while not np.array_equal(centers, nCenters):

			if flag:
				cCentroid = {}
				centers = nCenters
				count = 1
				for center in centers:
					cCentroid["Cluster" + str(count)] = center
					count=count + 1

				cPoints = {}
				intermList = []
				for key,val in cCentroid.items():
					intermList.append(list(val))
					cPoints[key] = intermList
					intermList = []
			#assigning the data point to clusters with smalled euclidean distance	
			for data in dataFile:
				if data not in centers:
					minimum = sys.maxsize
					minCenter = sys.maxsize
					for center in centers:
						distCenter = distance.euclidean(data,center)
						if distCenter < minimum:
							minimum = distCenter
							minCenter = center
					for key, val in cCentroid.items():
						if str(val) == str(minCenter):
							if key not in cPoints.keys():
								dPoints = []
								dPoints.append(list(data))
								cPoints[key] = dPoints
							else:
								dPoints = cPoints[key]
								dPoints.append(list(data))
								cPoints[key] = dPoints
			nCenters = []
			for key, val in cPoints.items():
				result1 = np.mean(val, axis=0)
				nCenters.append(result1)
			nCenters = np.array(nCenters)
			flag = True
		index = 0
		for key,val in cCentroid.items():
			cCentroid[key] = centers[index]
			index += 1
		#calculating objective function value
		ss = 0
		for key, val in cPoints.items():
			cKValue1 = list(cCentroid.get(key))
			for value in val:
				dist1 = distance.euclidean(value, cKValue1)
				distanceSquared = math.pow(dist1, 2)
				ss += distanceSquared
		J.append(ss)
	#plotting the graph
	print("Object Function = %s" % J)
	s1graphPlot.title("K-Means Strategy 1")
	K = [k for k in range(2,11)]
	l = 'Initialization = ' + str(initial+1)
	s1graphPlot.xlabel('Number of Clusters = k')
	s1graphPlot.ylabel('Objective Function = J(k)')
	s1graphPlot.plot(K, J, x, marker='o', label=l)
	s1graphPlot.legend()
	print("Initial Objective Function value = " + str(initial+1))
	print("K: "+str(K))
	print("\n")
	print("J(K): "+str(J))
	print("\n")
	s1graphPlot.show()
	x = 'red'
	initial += 1

s1graphPlot.show()