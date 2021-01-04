
# Zero Shot Learning

The running environment requirements for running this code is as follows:
	Python 2.7
	Tensorflow 1.0.0
	Keras 2.2.4
	Nvidia Driver Version 384.183
	CUDA Version 9.0
Dataset url : https://cvml.ist.ac.at/AwA2/

Data Preprocessing:
1. Download data.zip to get the image dataset used, store it in a folder
2. Open python file 'DataLoader_AWA2.py' and edit the dataset name as the folder where data is stored.
3. Save the changes and run DataLoader_AWA2.py by using command : python DataLoader_AWA2.py
4. After runnning this file several .h5 files will be generated for AWA1/AW2 datasets. 
5. We can either run AWA1 or AWA2 at a time. We have to change 'dataset' variable to change input dataset
This step will train and save models as .h5 extensions which contains feature information w.r.t class label and wordtoVec class.


Training and Testing Data:
1. For training dataset, open WAV_Fusion_train.py 
2. Set variable 'dataroot' = 'Path of root folder of all datasets'.
3. Set variable 'datasets' = list of name of all datasets for whom you want to train.
For AWA2, keep is same as: datasets = ['AWA2'] #line 705
4. Run file WAV_Fusion_train.py using command : python WAV_Fusion_train.py
5. Accuray can seen on print screen.
6. Predicted class labels will be stored at './results_testModel_AWA2.csv' or './results_testModel_AWA2.csv'