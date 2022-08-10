"""
Author : Abhishek Mohabe
Course : CSE 575
"""
import scipy.io, numpy as np,random, sys,math, matplotlib.pyplot as s2graphPlot
from scipy.spatial import distance

dataFile = scipy.io.loadmat("AllSamples.mat")['AllSamples']
dataFileDimensions = dataFile.shape
print("K-Means Strategy 2" + "\n")

initial = 0
x = 'c'
while initial <= 1:
	J = []
	for k in range(2,11):
		centers = []
		cCentroid = {}
		cPoints = {}
		nCenters = []
		#generating 1 index randomly
		randIndices = np.random.choice(dataFileDimensions[0], 1, replace = False)
		centers.append(list(dataFile[randIndices][0]))
		#assigning rest of the centers basis the strategy 2
		while len(centers) < k:

			maximum = -sys.maxsize - 1
			mCenter = -sys.maxsize - 1
			for i in dataFile:
				distCenter = 0
				for j in centers:
					if list(i) not in centers:
						distCenter += distance.euclidean(i, j)
				distCenter = distCenter/len(centers)
				if distCenter > maximum:
					maximum = distCenter
					mCenter = i
			centers.append(list(mCenter))

		for i in range(len(randIndices)):
			nCenters.append(None)
		nCenters = np.array(nCenters)
		count = 1
		for center in centers:
			cCentroid["Cluster" + str(count)] = center
			count = count + 1
		centers = np.array(centers)

		iList1 = []
		for key,val in cCentroid.items():
			iList1.append(list(val))
			cPoints[key] = iList1
			iList1 = []
		#assigning data points until centroid value remains same	
		flag = False
		while not np.array_equal(centers,nCenters):
			if flag:
				cCentroid = {}
				centers = nCenters
				count = 1
				for center in centers:
					cCentroid["Cluster" + str(count)] = center
					count = count + 1
				cPoints = {}
				iList1 = []
				for key,val in cCentroid.items():
					iList1.append(list(val))
					cPoints[key] = iList1
					iList1 = []
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
						if list(val) == list(minCenter):
							if key not in cPoints.keys():
								dPoints = []
								dPoints.append(list(data))
								cPoints[key] = dPoints
							else:
								dPoints = cPoints[key]
								dPoints.append(list(data))
								cPoints[key]=dPoints
			nCenters = []
			for key,val in cPoints.items():
				result1 = np.mean(val,axis=0)
				nCenters.append(result1)
			nCenters = np.array(nCenters)
			flag = True
		index = 0
		for key, val in cCentroid.items():
			cCentroid[key] = centers[index]
			index += 1
		#calculating objective function value
		ss = 0
		for key, val in cPoints.items():
			cKValue1 = list(cCentroid.get(key))
			for value in val:
				ss += math.pow(distance.euclidean(value,cKValue1),2)
		J.append(ss)
	#plotting the graph
	print("Objective function = %s" % J)
	K = [k for k in range(2,11)]
	l = 'Initialization = '+str(initial+1)
	s2graphPlot.title("K-Means Strategy 2")
	s2graphPlot.ylabel('Objective Function = J(k)')
	s2graphPlot.xlabel('Number of Clusters = k')
	s2graphPlot.plot(K, J, x, marker='o', label=l)
	s2graphPlot.legend()
	print("Initial Objective Function value = " + str(initial+1))
	print("K:" + str(K))
	print("\n")
	print("J(K):" + str(J))
	print("\n")
	s2graphPlot.show()
	x = 'yellow'
	initial += 1
s2graphPlot.show()