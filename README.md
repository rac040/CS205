# CS205

compile_data.py - 
	Compiles sessions data into single csv files for feature extraction. Assumes that sensor data is still compressed and in the same format as originally stored on watch. It expects the following file directory structure:
	TRAIN_DIR = PATH + '/Data/Sessions/'
	TEST_DIR = PATH + '/Data/test_data/'
	TRAIN_OUTPUT_DIR = PATH + '/Data/Compiled/Train/'
	TEST_OUTPUT_DIR = PATH + '/Data/Compiled/Test/'

features.py - 
	Extracts feature data from compiled data. Assumes followinf file directory:
	TRAIN_IN_DIR = PATH + '/Data/Compiled/Train/'
        TEST_IN_DIR = PATH + '/Data/Compiled/Test/'
	OUTPUT = PATH + '/Data/Processed/'

lgbm_train.py - 
	Trains and tests model based on data accumulated using features.py. Assumes the following directory path:
	IN_DIR = PATH + '/Data/Processed/'
	OUTPUT = PATH + '/Data/Results/'


To Run:
#First create appropriate file directories and store data appropriately
	python features.py
	python lgbm_train.py