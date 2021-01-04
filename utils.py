import math
import numpy as np
import pdb
from numpy import *

def getLabelPredictions(predictions, attributes,classes, similarity="euclidean"):
	if similarity == "cosine":
		bili = np.dot(predictions, np.transpose(attributes))
		preds = np.argmax(bili, axis=1)
	elif similarity == "euclidean":
		#pdb.set_trace()
		preds = np.zeros(predictions.shape[0], )
		for ii in range(predictions.shape[0]):
			curSample = predictions[ii]
			diff = tile(curSample, (attributes.shape[0], 1))-attributes
			squaredDiff = diff ** 2 # squared for the subtract  	
			squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row  
			distance = squaredDist ** 0.5 
			preds[ii] = argmin(distance)
	# pdb.set_trace()
	preds.tolist()
	preds = [classes[int(i)] for i in preds]
	return preds
	# 
	

def computePerClassAcc(trueClasses, predictions):
	uniqTrueClasses = np.unique(trueClasses)
	allAccs = [0]*len(uniqTrueClasses)
	for i in range(len(uniqTrueClasses)):
		curClass = uniqTrueClasses[i]
		sampleSet = trueClasses == curClass
		# pdb.set_trace()
		curClassPreds = [predictions[j] for j in range(len(predictions)) if sampleSet[j]]
		truePreds = curClassPreds == curClass
		acc = float(np.sum(truePreds))/np.sum(sampleSet)
		# pdb.set_trace()
		allAccs[i] = acc
	perClassAcc = np.mean(allAccs)
	return perClassAcc

def computeHScore():
	pass
