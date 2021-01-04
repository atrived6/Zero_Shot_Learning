from keras import optimizers, regularizers
from keras.layers import Dense, Dot, Dropout, Activation, Input
from keras.layers.merge import Subtract
from keras.layers.core import Lambda, Reshape, RepeatVector
from keras.models import Model
from keras.callbacks import Callback
import pdb
import sys
import utils
import numpy as np
import datetime
import json
from utils import getLabelPredictions, computePerClassAcc
# import cv2 
import random
#import h5py
#from customLossFunctions import tripletLoss, bilinearLoss
from scipy import io as sio
from scipy.spatial.distance import cdist
import ast
import keras.backend as K
import csv
#from datetime import datetime
import os
import time
import h5py as hp
#from gensim.models import KeyedVectors
# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

'''
Notes:
- weight initialization: glorot uniform
- model implemented with Keras is working fine as per baseline_deep_emebedding2.py

Pending:
- check activation
- Cross check the code and make sure your haven't made any mistakes
- stability issue.
-  


Issues:

- replicate results of Good, Bad, Ugly paper
'''
'''
def getWord2Vecs(classNames):
    model = KeyedVectors.load_word2vec_format('./word2Vec/GoogleNews-vectors-negative300.bin', binary=True)
    vecs = np.zeros((len(classNames), 300))
    exclude = []
    newClassNames =[]
    for i in range(len(classNames)):
        try:
            curClass = classNames[i]
            curClass = curClass.replace('+', '_')
            vecs[i] = model[curClass]
            newClassNames.append(curClass)
        except:
            # pdb.set_trace()
            # vecs[i] = np.ones((300,))*-100
            exclude.append(i+1)
            print('skipped the class: '+curClass)
            pdb.set_trace()
    return vecs, exclude, newClassNames	
'''
def loadData():

	matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
	# numpy array index starts from 0, matlab starts from 1
	trainval_loc = matcontent['trainval_loc'].squeeze() - 1
	test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
	attribute = matcontent['original_att'].T  # att
	classNames = matcontent['allclasses_names']
	classNames = [str(i[0][0]) for i in classNames]

	fTrain = hp.File(dataroot + '/' + dataset + '/' + 'train3.h5', 'r')
	fVal = hp.File(dataroot + '/' + dataset + '/' + 'val3.h5', 'r')
	fTest = hp.File(dataroot + '/' + dataset + '/' + 'test.h5', 'r')
	w2vMat = sio.loadmat(dataroot + "/" + dataset + '/word2vecs.mat')
	word2vecs = w2vMat['word2vecs']
	word2vecs = word2vecs.squeeze()
	exclude = w2vMat['exclude']
	newClassNames = w2vMat['newClassNames']

	features_train = fTrain['features'][:]
	attrVecs_train = fTrain['attrVecs'][:]
	labels_train = fTrain['labels'][:][:]
	labels_train = labels_train.squeeze() - 1
	trainClasses = np.unique(labels_train)
	trainAttributes = attribute[trainClasses]
	wordVecs_train = word2vecs[labels_train]
	trainWord2Vecs = word2vecs[trainClasses]
	totalTrainSamples = features_train.shape[0]

	features_val = fVal['features'][:]
	attrVecs_val = fVal['attrVecs'][:]
	labels_val = fVal['labels'][:]
	labels_val = labels_val.squeeze() - 1
	valClasses = np.unique(labels_val)
	valAttributes = attribute[valClasses]
	wordVecs_val = word2vecs[labels_val]
	valWord2Vecs = word2vecs[valClasses]
	totalValSamples = features_val.shape[0]

	features_test = fTest['features'][:]
	attrVecs_test = fTest['attrVecs'][:]
	labels_test = fTest['labels'][:]
	labels_test = labels_test.squeeze() - 1
	testClasses = np.unique(labels_test)
	#pdb.set_trace()
	testAttributes = attribute[testClasses]
	wordVecs_test = word2vecs[labels_test]
	testWord2Vecs = word2vecs[testClasses]
	totaltestSamples = features_test.shape[0]

	#pdb.set_trace()

	return [features_train, labels_train, attrVecs_train, trainClasses, trainAttributes, \
			wordVecs_train, trainWord2Vecs, totalTrainSamples], \
		   [features_val, labels_train, attrVecs_val, valClasses, \
		valAttributes, wordVecs_val, valWord2Vecs, totalValSamples], \
		   [features_test, labels_test, attrVecs_test, testClasses, testAttributes, \
			wordVecs_test, testWord2Vecs, totaltestSamples], \
		   [attribute, classNames, word2vecs]

def makeData():
	global dataroot, dataset, image_embedding, class_embedding, dataSplit
	matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
	feature = matcontent['features'].T
	label = matcontent['labels'].astype(int).squeeze() - 1
        
	matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
	# numpy array index starts from 0, matlab starts from 1
	trainval_loc = matcontent['trainval_loc'].squeeze() - 1
	test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
	test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
	attribute = matcontent['original_att'].T # att
	classNames = matcontent['allclasses_names']
	classNames = [str(i[0][0]) for i in classNames]

	# word2vecs, exclude, newClassNames = getWord2Vecs(classNames)
	# sio.savemat(dataroot + "/" + dataset + '/word2vecs.mat', {'word2vecs':word2vecs, 'exclude':exclude, 'newClassNames':newClassNames})
	w2vMat = sio.loadmat(dataroot + "/" + dataset + '/word2vecs.mat')
	word2vecs = w2vMat['word2vecs']
	exclude = w2vMat['exclude']
	newClassNames = w2vMat['newClassNames']

	train_label = label[trainval_loc]
	train_loc = [trainval_loc[i] for i in range(len(trainval_loc)) if train_label[i] not in exclude]

	train_vis = feature[train_loc]
	train_label = label[train_loc]
	train_attr = attribute[train_label]
	trainClasses = np.unique(train_label)
	trainAttributes = attribute[trainClasses]
	train_wordVec = word2vecs[train_label]
	trainWord2Vecs = word2vecs[trainClasses]
	totalTrainSamples = train_vis.shape[0]

	test_unseen_label = label[test_unseen_loc]
	test_unseen_loc = [test_unseen_loc[i] for i in range(len(test_unseen_loc)) if test_unseen_label[i] not in exclude]

	test_unseen_vis = feature[test_unseen_loc]
	test_unseen_label = label[test_unseen_loc]
	test_unseen_attr = attribute[test_unseen_label]
	testUnseenClasses = np.unique(test_unseen_label)
	testUnseenAttributes = attribute[testUnseenClasses]
	test_unseen_wordVec = word2vecs[test_unseen_label]
	testUnseenWord2Vecs = word2vecs[testUnseenClasses]
	totaltestUnseenSamples = test_unseen_vis.shape[0]

	test_seen_label = label[test_seen_loc]
	test_seen_loc = [test_seen_loc[i] for i in range(len(test_seen_loc)) if test_seen_label[i] not in exclude]

	test_seen_vis = feature[test_seen_loc]
	test_seen_label = label[test_seen_loc]
	test_seen_attr = attribute[test_seen_label]
	testSeenClasses = np.unique(test_seen_label)
	testSeenAttributes = attribute[testSeenClasses]
	test_seen_wordVec = word2vecs[test_seen_label]
	testSeenWord2Vecs = word2vecs[testSeenClasses]
	totaltestSeenSamples = test_seen_vis.shape[0]

	return [train_vis, train_label, train_attr, trainClasses, trainAttributes, \
	train_wordVec, trainWord2Vecs, totalTrainSamples], \
		[test_unseen_vis, test_unseen_label, test_unseen_attr, testUnseenClasses, \
		testUnseenAttributes, test_unseen_wordVec, testUnseenWord2Vecs, totaltestUnseenSamples], \
		[test_seen_vis, test_seen_label, test_seen_attr, testSeenClasses, testSeenAttributes, \
		test_seen_wordVec, testSeenWord2Vecs, totaltestSeenSamples], \
		[attribute, classNames, word2vecs]

def data_generator(dataset='Train', batch_size= 50, phase="Training", shuffle=True):
	""" A simple data iterator """
	global allData
	batch_idx = 0
	while True:
		if dataset == "Train":
			x = allData[0][0]
			y = allData[0][1]
			att = allData[0][5]
		elif dataset == "Validation":
			x = allData[1][0]
			y = allData[1][1]
			att = allData[1][5]
		elif dataset == "Test":
			x = allData[2][0]
			y = allData[2][1]
			att = allData[2][5]
		else:
			print("Enter either Train/Test/Valid for the dataset parameter")
		idxs = np.arange(0, len(x))
		if shuffle:
			np.random.shuffle(idxs)
		shuf_visual = x[idxs]
		shuf_att = att[idxs]
		shuff_y = y[idxs]
		batch_count = x.shape[0]// batch_size
		# pdb.set_trace()
		for batch_idx in range(batch_count):
			visual_batch = shuf_visual[batch_idx*batch_size:(batch_idx + 1)*batch_size]
			att_batch = shuf_att[batch_idx*batch_size:(batch_idx + 1)*batch_size]
			label_batch = shuff_y[batch_idx*batch_size:(batch_idx + 1)*batch_size]
			if phase == "Training":
				yield [att_batch], [visual_batch]
			else:
				yield [att_batch], [visual_batch, label_batch]

def test(phase="Training", batch_size=1):
	'''
	Print train per-class-acc
	Print val per-class acc
	Print test-unseen per-class acc ---- ZSL per-class acc
	Print test-seen + un-seen per-class acc ---- GZSL per-class acc
	Print H score using 
	'''
	global allData
	[train_vis, train_label, train_attr, trainClasses, trainAttributes, \
	train_wordVec, trainWord2Vecs, totalTrainSamples], \
		[valid_vis, valid_label, valid_attr, validClasses, \
		validAttributes, valid_wordVec, testValidWord2Vecs, totalValidSamples], \
		[test_vis, test_label, test_attr, testClasses, testAttributes, \
		test_wordVec, testWord2Vecs, totaltestSamples], \
		[attribute, classNames, word2vecs] = allData
	metrics={}
	trAttrPred = bilinear.predict(trainWord2Vecs, batch_size)
	trainPreds = getLabelPredictions(train_vis, trAttrPred, trainClasses)
	perClassAccTrain = computePerClassAcc(train_label, trainPreds)
	metrics['perClassAccTrain']=perClassAccTrain
	print("Extracted Training Predictions")
 
	valAttrPred = bilinear.predict(testValidWord2Vecs, batch_size)
	validPreds = getLabelPredictions(valid_vis, valAttrPred, validClasses)
	perClassAccValid = computePerClassAcc(valid_label, validPreds)
	metrics['perClassAccValid']=perClassAccValid
	print('Extracted Validation Prediction')

	testUnSeenAttrPred = bilinear.predict(testWord2Vecs, batch_size)
	testUnseenPreds = getLabelPredictions(test_vis, testUnSeenAttrPred, testClasses)
	perClassAccTestUnseen_ZSL = computePerClassAcc(test_label, testUnseenPreds)
	metrics['perClassAccTestUnseen_ZSL']=perClassAccTestUnseen_ZSL
	print("Extracted Test Unseen Predictions")

	return metrics

class saveWeights(Callback):
    def __init__(self):
        self.count = 0
        # self.valLoss=[]

    def on_train_begin(self, logs={}):
        test()
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        # self.valLoss.append(logs['val_loss'])
        # pdb.set_trace()
        bilinear.save_weights("./weights/bil_vis_attr_"+str(self.count)+'.h5')
        # if self.count % 5 == 0 and self.count > 0:
        #     # pdb.set_trace()
        #     metrics = testing("./weights/bil_vis_attr_"+str(self.count)+'.h5')
        #     print(metrics)
        self.count = self.count + 1
        # pass

    def on_train_end(self, logs={}):
        pass

def makeModelWV():
	global batch_size, nAttributes, nVis, reg

	# visual = Input(shape=(nVis,))
	w2v_feature = Input(shape=(nAttributes,))

	dense1 = Dense(1600, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(w2v_feature)
	activ1 = Activation('relu')(dense1)

	dense2 = Dense(nVis, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(activ1)
	activ2 = Activation('relu')(dense2)

	bilinear = Model(inputs=[w2v_feature], outputs=[activ2])
	#predModel = Model(inputs = [att_feature], outputs=[activ2])
	#return bilinear, predModel
	return bilinear

def makeModel():
	global batch_size, nAttributes, nVis, reg

	# visual = Input(shape=(nVis,))
	att_feature = Input(shape=(nAttributes,))

	dense1 = Dense(1600, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(att_feature)
	activ1 = Activation('relu')(dense1)

	dense2 = Dense(nVis, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(activ1)
	activ2 = Activation('relu')(dense2)

	bilinear = Model(inputs=[att_feature], outputs=[activ2])
	#predModel = Model(inputs = [att_feature], outputs=[activ2])
	#return bilinear, predModel
	return bilinear


def testing(loadWeights, phase="Testing"):
	# predModel.load_weights(loadWeights)
	metrics = test(phase)
	return metrics
	
def training():
	global bilinear, epochs, allData, batch_size, MODEL_DIR
	
	totalTrainSamples = allData[0][7]

	saveweights = saveWeights()
	bilinear.fit_generator(
			data_generator(),
			steps_per_epoch=totalTrainSamples//batch_size,
			epochs=epochs,
			verbose=1,
			validation_data=None,
			callbacks=[saveweights])
	# bestPerformanceWeights = saveweights.valLoss.index(min(saveweights.valLoss))
	# bestPerformanceWeights = len(saveweights.valLoss)-1
	cmd = "cp ./weights/bil_vis_attr_"+str(epochs-1)+".h5 "+MODEL_DIR
	os.system(cmd)
	return None
	
def run_validation():
	global bilinear, epochs, allData, batch_size, MODEL_DIR

	totalValSamples = allData[1][7]

	saveweights = saveWeights()
	bilinear.fit_generator(
	data_generator(dataset='Validation'),
	steps_per_epoch=totalValSamples // batch_size,
	epochs=epochs,
	verbose=1,
	validation_data=None,
	callbacks=[saveweights])
	# bestPerformanceWeights = saveweights.valLoss.index(min(saveweights.valLoss))
	# bestPerformanceWeights = len(saveweights.valLoss)-1
	cmd = "cp ./weights/bil_vis_attr_" + str(epochs - 1) + ".h5 " + MODEL_DIR
	os.system(cmd)
	return None
	
	# bilinear.save_weights(MODEL_DIR)

if __name__=="__main__":

	
	comments = "Testing Stability for deep embedding model implemented in Keras"
	batch_size = 50
	epochs = 60
	dataroot = './data'
	# datasets = ['AWA2', 'AWA1', 'CUB', 'SUN', 'APY']
	datasets = ['AWA1']
	image_embedding = 'res101'
	class_embedding = 'att' # original_att
	#nAttributesVec=[85, 85, 312, 102, 64]
	nAttributesVec=[300, 300, 300, 300, 300]
	nVis = 2048
	resultsFileName = "./allResults_WV_AWA1.csv"
	# margin=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	margin=[0.1]
	Embeddings="word2Vecs"
	regs=[0.001]
	# regs = [0.001, 0.01, 0.05, 0.1, 0.5]
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	with open(resultsFileName, mode='a') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(['TimeStamp', 'Dataset', 'Split Number', 'Embeddings', 'Run Number', 'margin', 'perClassAccTrain', 'perClassAccVal_ZSL', 'perClassAccVal_GZSL', 'perClassAccTestUnseen_ZSL', \
			'perClassAccTestUnseen_GZSL', 'perClassAccTestSeen_Supervised_Learning', 'ValLoss','regularizer','Comments'])

	for k in range(len(datasets)):
		for i in range(0, len(regs)): #Regularizer
			for j in range(len(margin)):
				for l in range(1, 6): #Run Number
					K.clear_session()  #https://github.com/keras-team/keras/issues/5345
					dataSplit = 1
					reg = regs[i]
					nAttributes = nAttributesVec[k]
					dataset=datasets[k]
					#bilinear, predModel = makeModel()
					bilinear = makeModelWV()
					bilinear.compile(loss=['mean_squared_error'],optimizer=optimizers.Adam(lr=0.0001))
					# pdb.set_trace()
					# bilinear.set_weights()
					allData = loadData()
					#pdb.set_trace()
					#time = int(datetime.timestamp(datetime.now()))
					timestr = time.strftime("%d-%m-%Y-%H:%M:%S")
					MODEL_DIR = "./weights/final_bil_vis_attr_"+str(timestr)+".h5"
					valL = training()
					metrics = testing(MODEL_DIR)
					print(metrics)
					with open(resultsFileName, mode='a') as csv_file:
						writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
						writer.writerow([time, dataset, None, Embeddings, l, margin[j], metrics['perClassAccTrain'], None,\
						None, metrics['perClassAccTestUnseen_ZSL'], None, None, \
						valL, reg, comments])
		# print allResults
