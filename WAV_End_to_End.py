

from keras import optimizers, regularizers
from keras.layers import Dense, Dot, Dropout, Activation, Input, Concatenate, Add, Conv1D, GlobalMaxPooling1D
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
import h5py
#from customLossFunctions import tripletLoss, bilinearLoss
from scipy import io as sio
from scipy.spatial.distance import cdist
import ast
import keras.backend as K
import csv
from datetime import datetime
import h5py as hp
import os
import time
#from gensim.models.keyedvectors import KeyedVectors
#from gensim.models import KeyedVectors

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_fraction = 0.6
# pdb.set_trace()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def generateAverageWordVectors( wordVectors, vectorWeights):

    return vectorWeights.dot(wordVectors)

def generateAttributeNameVectorPerClass(wordVectors_attr, wordVectors_labels, vectorWeights):

    vector_dict = {}

    for label in range(vectorWeights.shape[0]):
	attr_vec = np.zeros(shape=(86, 300))
	attr_vec[0] = wordVectors_labels[label]
	for attribute in range(vectorWeights.shape[1]):
	    attr_vec[attribute+1] =  wordVectors_attr[attribute] * vectorWeights[label][attribute]

	#pdb.set_trace()
	attr_vec = np.array(attr_vec)
	vector_dict[label] = attr_vec

    return vector_dict


def loadData():

    # Load the estimated attribute vectors for each of the images
    '''
    predAttributeContent = sio.loadmat(dataroot + "/" + dataset + "/" + "attr_est.mat")
    predAttr_train = predAttributeContent['S_est_tr']
    predAttr_val = predAttributeContent['S_est_val']
    predAttr_test  = predAttributeContent['S_est_te']
    '''
    # Load predicate matrix : 50 X 85
    predicate = sio.loadmat(dataroot + "/" + dataset + "/" + "predicateMatrix.mat")
    predicateMatrix = predicate['predicateMatrix']

    # Load Attribute Name Vectors : 85 X 300
    attributeContent = sio.loadmat(dataroot + "/" + dataset + "/" + "attributeVectors.mat")
    attributeNameVectors = attributeContent['attributeVectors']


    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
	# numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    attribute = matcontent['original_att'].T  # att
    classNames = matcontent['allclasses_names']
    classNames = [str(i[0][0]) for i in classNames]

    fTrain = hp.File(dataroot + '/' + dataset + '/' + 'train1.h5', 'r')
    fVal = hp.File(dataroot + '/' + dataset + '/' + 'val1.h5', 'r')
    fTest = hp.File(dataroot + '/' + dataset + '/' + 'test.h5', 'r')
    w2vMat = sio.loadmat(dataroot + "/" + dataset + '/word2vecs.mat')
    word2vecs = w2vMat['word2vecs']
    word2vecs = word2vecs.squeeze()
    exclude = w2vMat['exclude']
    newClassNames = w2vMat['newClassNames']
    
    # Create Averaged Attribute Name Vectors
    averagedAttributeNameVector = generateAttributeNameVectorPerClass(attributeNameVectors, word2vecs, predicateMatrix) # Dimensions: 50 X 85 X 300

    features_train = fTrain['features'][:]
    attrVecs_train = fTrain['attrVecs'][:]
    labels_train = fTrain['labels'][:][:]
    #attrNameVecs_train = generateAverageWordVectors(attributeNameVectors, predAttr_train)
    attrNameVecs_train = np.zeros((features_train.shape[0],86,300))
    labels_train = labels_train.squeeze() - 1
    for index in range(attrNameVecs_train.shape[0]):
	attrNameVecs_train[index] = averagedAttributeNameVector[labels_train[index]]
    trainClasses = np.unique(labels_train)
    trainAttributes = attribute[trainClasses]
    wordVecs_train = word2vecs[labels_train]
    trainWord2Vecs = word2vecs[trainClasses]
    totalTrainSamples = features_train.shape[0]
    trainAttributeNameVecs = list()
    for train_label in trainClasses:
        trainAttributeNameVecs.append(averagedAttributeNameVector[train_label])
    
    features_val = fVal['features'][:]
    attrVecs_val = fVal['attrVecs'][:]
    labels_val = fVal['labels'][:]
    labels_val = labels_val.squeeze() - 1
    #attrNameVecs_valid = generateAverageWordVectors(attributeNameVectors, predAttr_val)
    attrNameVecs_valid = np.zeros((features_val.shape[0],86,300))
    for index in range(attrNameVecs_valid.shape[0]):
        attrNameVecs_valid[index] = averagedAttributeNameVector[labels_val[index]]
    valClasses = np.unique(labels_val)
    valAttributes = attribute[valClasses]
    wordVecs_val = word2vecs[labels_val]
    valWord2Vecs = word2vecs[valClasses]
    totalValSamples = features_val.shape[0]
    validAttributeNameVecs = list()
    for val_label in valClasses:
        validAttributeNameVecs.append(averagedAttributeNameVector[val_label])
    
    features_test = fTest['features'][:]
    attrVecs_test = fTest['attrVecs'][:]
    labels_test = fTest['labels'][:]
    labels_test = labels_test.squeeze() - 1
    #attrNameVecs_test = generateAverageWordVectors(attributeNameVectors, predAttr_test)
    attrNameVecs_test = np.zeros((features_test.shape[0], 86,300))
    for index in range(attrNameVecs_test.shape[0]):
	attrNameVecs_test[index] = averagedAttributeNameVector[labels_test[index]]
    testClasses = np.unique(labels_test)
    #pdb.set_trace()
    testAttributes = attribute[testClasses]
    wordVecs_test = word2vecs[labels_test]
    testWord2Vecs = word2vecs[testClasses]
    totaltestSamples = features_test.shape[0]
    testAttributeNameVecs = list()
    for test_label in testClasses:
        testAttributeNameVecs.append(averagedAttributeNameVector[test_label])
 
    #pdb.set_trace()
    return [features_train, labels_train, attrVecs_train, trainClasses, trainAttributes, \
			wordVecs_train, trainWord2Vecs, trainAttributeNameVecs, totalTrainSamples, attrNameVecs_train], \
		   [features_val, labels_train, attrVecs_val, valClasses, \
			valAttributes, wordVecs_val, valWord2Vecs, validAttributeNameVecs, totalValSamples, attrNameVecs_valid], \
		   [features_test, labels_test, attrVecs_test, testClasses, testAttributes, \
			wordVecs_test, testWord2Vecs, testAttributeNameVecs, totaltestSamples, attrNameVecs_test], \
		   [attribute, classNames, word2vecs, averagedAttributeNameVector]
'''
    return [features_train, labels_train, attrVecs_train, trainClasses, trainAttributes, \
			wordVecs_train, trainWord2Vecs, trainAttributeNameVecs, totalTrainSamples, attrNameVecs_train], \
		   [features_val, labels_train, attrVecs_val, valClasses, \
			valAttributes, wordVecs_val, valWord2Vecs, validAttributeNameVecs, totalValSamples, attrNameVecs_valid], \
		   [features_test, labels_test, attrVecs_test, testClasses, testAttributes, \
			wordVecs_test, testWord2Vecs, testAttributeNameVecs, totaltestSamples, attrNameVecs_test], \
		   [attribute, classNames, word2vecs, averagedAttributeNameVector]
'''

def data_generator(dataset='Train', batch_size=50, phase="Training", shuffle=True):
    """ A simple data iterator """
    global allData
    batch_idx = 0
    while True:
        if dataset == "Train":
            x = allData[0][0]
            y = allData[0][1]
            attr = allData[0][2]
	    a2v = allData[0][9]
            w2v = allData[0][5]
        elif dataset == "Validation":
            x = allData[1][0]
            y = allData[1][1]
            attr = allData[1][2]
	    a2v = allData[1][9]
            w2v = allData[1][5]
        elif dataset == "Test":
            x = allData[2][0]
            y = allData[2][1]
            attr = allData[2][2]
	    a2v = allData[2][9]
            w2v = allData[2][5]
        else:
            print("Enter either Train/Test/Valid for the dataset parameter")
        idxs = np.arange(0, len(x))
        if shuffle:
            np.random.shuffle(idxs)
        shuf_visual = x[idxs]
        shuf_attr = attr[idxs]
	shuf_a2v = a2v[idxs]
        shuf_w2v = w2v[idxs]
        shuff_y = y[idxs]
        batch_count = x.shape[0] // batch_size
        # pdb.set_trace()
        for batch_idx in range(batch_count):
            visual_batch = shuf_visual[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            w2v_batch = shuf_w2v[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            a2v_batch = shuf_a2v[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            attr_batch = shuf_attr[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            label_batch = shuff_y[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            if phase == "Training":
                yield [attr_batch],  [visual_batch]
            else:
                yield [attr_batch],  [visual_batch, label_batch]

def test(phase="Training", batch_size=1):
    # Print train per-class-acc
    # Print val per-class acc
    # Print test-unseen per-class acc ---- ZSL per-class acc
    # Print test-seen + un-seen per-class acc ---- GZSL per-class acc
    # Print H score using
    global allData, sim
    [train_vis, train_label, train_attr, trainClasses, trainAttributes, \
     train_wordVec, trainWord2Vecs, trainAttributeNameVecs,totalTrainSamples, attrNameVecs_train], \
    [valid_vis, valid_label, valid_attr, validClasses, \
     validAttributes, valid_wordVec, validWord2Vecs, validAttributeNameVecs, totalValSamples, attrNameVecs_valid], \
    [test_vis, test_label, test_attr, testClasses, testAttributes, \
     test_wordVec, testWord2Vecs, testAttributeNameVecs, totaltestSamples, attrNameVecs_test], \
    [attribute, classNames, word2vecs, averageAttributeNameVectors] = allData
    metrics = {}
    
    #_, trVisPred = bilinear.predict([np.array(trainWord2Vecs),np.array(trainAttributeNameVecs)], batch_size)
    trVisPred = bilinear.predict(trainAttributes, batch_size)
    trainPreds = getLabelPredictions(train_vis, trVisPred, trainClasses, sim)
    perClassAccTrain = computePerClassAcc(train_label, trainPreds)
    metrics['perClassAccTrain'] = perClassAccTrain
    print("Extracted Training Predictions")

    #_, valVisPred = bilinear.predict([np.array(validWord2Vecs),np.array(validAttributeNameVecs)], batch_size)
    valVisPred = bilinear.predict(validAttributes, batch_size)
    valPreds = getLabelPredictions(valid_vis, valVisPred, validClasses, sim)
    perClassAccVal = computePerClassAcc(valid_label, valPreds)
    metrics['perClassAccVal'] = perClassAccVal
    print('Extracted Validation Predictions')

    #_, teVisPred  = bilinear.predict([np.array(testWord2Vecs),np.array(testAttributeNameVecs)], batch_size)
    teVisPred  = bilinear.predict(testAttributes, batch_size)
    testPreds = getLabelPredictions(test_vis, teVisPred, testClasses, sim)
    perClassAccTest_ZSL = computePerClassAcc(test_label, testPreds)
    metrics['perClassAccTestUnseen_ZSL'] = perClassAccTest_ZSL
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
        bilinear.save_weights("./weights/bil_vis_attr_" + str(self.count) + '.h5')
        # if self.count % 5 == 0 and self.count > 0:
        #     # pdb.set_trace()
        #     metrics = testing("./weights/bil_vis_attr_"+str(self.count)+'.h5')
        #     print(metrics)
        self.count = self.count + 1
        # pass

    def on_train_end(self, logs={}):
        pass

def makeTestModel():

    global batch_size, nW2V, nAttributes, nVis, reg

    #a2v = Input(shape=(86, nW2V))
    att_feature = Input(shape=(nAttributes,))
    
    dense1 = Dense(512, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(att_feature)
    activ1 = Activation('relu')(dense1)
 
    dense2 = Dense(1600, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(activ1)
    activ2 = Activation('relu')(dense2)
 
    dense3 = Dense(nVis, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(activ2)
    activ3 = Activation('relu')(dense3)
 
    bilinear = Model(inputs=[att_feature], outputs=[activ3])

    return bilinear

def makeModelCNN():
  
    #w2v = Input(shape=(nW2V,))
    a2v = Input(shape=(86, nW2V))
    
    # Class label word2Vec vectors
    #dense1_w2v = Dense(256, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(w2v)
    #activ1_w2v = Activation('relu')(dense1_w2v)

    #dense2_w2v = Dense(128, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(activ1_w2v)
    #activ2_w2v = Activation('relu')(dense2_w2v)

    #Model1 = Model(inputs=w2v, outputs=activ2_w2v)

    #attribute name word2Vec vector feature extraction using CNN layers
    conv1_a2v = Conv1D(256, 5, activation='relu')(a2v)

    #dense1_a2v = Dense(256, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(a2v)
    #activ1_a2v = Activation('relu')(dense1_a2v)
    max_pool1D = GlobalMaxPooling1D()(conv1_a2v)

    dense2_a2v = Dense(128, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(max_pool1D)
    activ2_a2v = Activation('relu')(dense2_a2v)

    #Model2 = Model(inputs=a2v, outputs=activ2_a2v)


    #combined = Add()([Model1.output, Model2.output])

    dense3 = Dense(nAttributes, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(activ2_a2v)
    activ3 = Activation('relu')(dense3)

    dense4 = Dense(512, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(activ3)
    activ4 = Activation('relu')(dense4)

    dense5 = Dense(1600, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(activ4)
    activ5 = Activation('relu')(dense5)

    dense6 = Dense(nVis, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(activ5)
    activ6 = Activation('relu')(dense6)

    bilinear = Model(inputs=[a2v], outputs=[activ3, activ6])

    return bilinear

def makeModelFusion2():
    global batch_size, nW2V, nAttributes, nVis, reg

    w2v = Input(shape=(nW2V,))
    a2v = Input(shape=(nW2V,))
    attr = Input(shape=(nAttributes,))

    dense1_w2v = Dense(256, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2
(reg))(w2v)
    activ1_w2v = Activation('relu')(dense1_w2v)

    dense2_w2v = Dense(128, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2
(reg))(activ1_w2v)
    activ2_w2v = Activation('relu')(dense2_w2v)

    Model1 = Model(inputs=w2v, outputs=activ2_w2v)

    dense1_a2v = Dense(256, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2
(reg))(a2v)
    activ1_a2v = Activation('relu')(dense1_a2v)

    dense2_a2v = Dense(128, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2
(reg))(activ1_a2v)
    activ2_a2v = Activation('relu')(dense2_a2v)

    Model2 = Model(inputs=a2v, outputs=activ2_a2v)

    combined1 = Add()([Model1.output, Model2.output])

    dense3 = Dense(85, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(combined1)
    activ3 = Activation('relu')(dense3)

    combined2 = Add()([attr, activ3])

    dense4 = Dense(512, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg))(combined2)
    activ4 = Activation('relu')(dense4)

    dense5 = Dense(1600, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2
(reg))(activ4)
    activ5 = Activation('relu')(dense5)

    dense6 = Dense(nVis, kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2
(reg))(activ5)
    activ6 = Activation('relu')(dense6)

    bilinear = Model(inputs=[w2v, a2v, attr], outputs=[activ6])

    return bilinear


def testing(loadWeights, phase="Testing"):
    # predModel.load_weights(loadWeights)
    metrics = test(phase)
    return metrics


def training():
    global bilinear, epochs, allData, batch_size, MODEL_DIR

    totalTrainSamples = allData[0][8]

    saveweights = saveWeights()
    bilinear.fit_generator(
        data_generator(),
        steps_per_epoch=totalTrainSamples // batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=None,
        callbacks=[saveweights])
    # bestPerformanceWeights = saveweights.valLoss.index(min(saveweights.valLoss))
    # bestPerformanceWeights = len(saveweights.valLoss)-1
    cmd = "cp ./weights/bil_vis_attr_" + str(epochs - 1) + ".h5 " + MODEL_DIR
    os.system(cmd)
    return None

def run_validation():
    global bilinear, epochs, allData, batch_size, MODEL_DIR

    totalValSamples = allData[1][8]

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

if __name__ == "__main__":

    #comments = "First Model: W to A is a two layer neural network, and A to V is an autoencoder network, all trained end-to-end; Loss weights are [1, 1]; Running to check stability"
    comments = "Testing Attribute to Visual Space Model"
    batch_size = 50
    epochs = 60
    dataroot = './data'
    # datasets = ['AWA2', 'AWA1', 'CUB', 'SUN', 'APY']
    datasets = ['AWA2']
    image_embedding = 'res101'
    class_embedding = 'att'  # original_att
    nAttributesVec = [85, 85, 312, 102, 64]
    #nAttributesVec = [64]
    nW2V_Vec = [300, 300, 300, 300, 300]
    nVis = 2048
    #resultsFileName = "./allResults_WAV_Fusion_predicate_add_APY.csv"
    resultsFileName = "./results_testModel_AWA2.csv"
    # margin=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    margin = [0.1]
    Embeddings = "word2Vecs"
    regs = [0.001]
    sim = "euclidean"  # cosine or euclidean
    # regs = [0.001, 0.01, 0.05, 0.1, 0.5]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    with open(resultsFileName, mode='a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['TimeStamp', 'Dataset', 'Split Number', 'Embeddings', 'Run Number', 'margin', 'perClassAccTrain',
             'perClassAccVal_ZSL', 'perClassAccVal_GZSL', 'perClassAccTestUnseen_ZSL', \
             'perClassAccTestUnseen_GZSL', 'perClassAccTestSeen_Supervised_Learning', 'ValLoss', 'regularizer',
             'similarity', 'Comments'])

    for k in range(len(datasets)):
        for i in range(0, len(regs)):  # Regularizer
            for j in range(len(margin)):
                for l in range(1, 6):  # Run Number
                    K.clear_session()  # https://github.com/keras-team/keras/issues/5345
                    dataSplit = 1
                    reg = regs[i]
                    nAttributes = nAttributesVec[k]
                    nW2V = nW2V_Vec[k]
                    dataset = datasets[k]
                    bilinear = makeTestModel()
                    bilinear.compile(loss=['mean_squared_error'],
                                     optimizer=optimizers.Adam(lr=0.0001), loss_weights=[1])
                    #bilinear.compile(loss=['mean_squared_error', 'mean_squared_error'],
                    #                 optimizer=optimizers.Adam(lr=0.0001), loss_weights=[1, 1])
                    # pdb.set_trace()
                    # bilinear.set_weights()
                    #allData = makeData()
                    allData = loadData()
                    # pdb.set_trace()
		    timestr = time.strftime("%d-%m-%Y-%H:%M:%S")
                    #time = int(datetime.timestamp(datetime.now()))
                    MODEL_DIR = "./weights/final_bil_vis_attr_" + str(timestr) + ".h5"
                    trainL = training()
		    valL = run_validation()
                    metrics = testing(MODEL_DIR)
                    print(metrics)
                    with open(resultsFileName, mode='a') as csv_file:
                        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(
                            [time, dataset, None, Embeddings, l, margin[j], metrics['perClassAccTrain'], None, \
                             None, metrics['perClassAccTestUnseen_ZSL'], None, None, \
                             valL, reg, sim, comments])


