from gensim.models import KeyedVectors
from string import digits
import re
import numpy as np
import pdb

def getAttributeList():
	file_path = "./data/AWA2/predicates.txt"
	attr_list = []
	with open(file_path) as fp:
		for line in fp:
			print line
			line = re.sub(r"\s+", "", line, flags=re.UNICODE)
			attr_list.append(line.translate(None, digits))

	return attr_list

def getAttributeNameVectors():

	attr_list = getAttributeList()
	print attr_list

	model = KeyedVectors.load_word2vec_format('./word2Vec/GoogleNews-vectors-negative300.bin', binary=True)
	vecs = np.zeros((len(attr_list), 300))

	#pdb.set_trace()

	for i in range(len(attr_list)):
		if attr_list[i] not in model.wv.vocab:
			print "Not found: " + attr_list[i]
			vecs[i] = np.random.normal(size=300)
		else:
			vecs[i] = model[attr_list[i]]

	return vecs

if __name__ == "__main__":

	attr_vecs = getAttributeNameVectors()
	print attr_vecs
