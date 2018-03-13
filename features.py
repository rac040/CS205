import os
import sys
PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import preprocessing

import csv
import gc
import time

#Run sensor data compilation
import compile_data

#Window size in seconds
WINDOW = 1
ACCURACY_THRESH = 2
TEST_SPLIT = 10
DATA_DIR = PATH + '/Data/Compiled/'

start = time.time()
print("\nStarting feature extraction...")

#Get acceleration data from file
accel_data = pd.read_csv(DATA_DIR + '1_android.sensor.accelerometer.data.csv',
            names=['timestamp_millsec', 'acc_force_x_axis', 'acc_force_y_axis',
                     'acc_force_z_axis', 'accuracy', 'label'],
            dtype={'timestamp_millsec': np.uint64,
                    'acc_force_x_axis': np.float32,
                    'acc_force_y_axis': np.float32,
                    'acc_force_z_axis': np.float32,
                    'accuracy': np.int8,
                    'label': 'category'})

#Convert milliseconds to seconds for our window evaluation
def get_window(x):
    if type(WINDOW) == int:
        time_sec = int(float(x) * float(pow(10,-3)))
    else:
        time_sec = float(x) * float(pow(10,-3))

    if time_sec % WINDOW == 0:
        return time_sec
    else:
        return time_sec - (time_sec % WINDOW)
accel_data['timestamp_sec'] = accel_data['timestamp_millsec'].map(get_window)
accel_data['resultant_acc'] = np.sqrt(np.power(accel_data['acc_force_x_axis'],2) + np.power(accel_data['acc_force_y_axis'],2) + np.power(accel_data['acc_force_z_axis'],2))
#Cleans all values not in accuracy threshold
accel_data = accel_data[accel_data.accuracy >= ACCURACY_THRESH]

#Get acceleration data from file
linear_accel_data = pd.read_csv(DATA_DIR + '10_android.sensor.linear_acceleration.data.csv',
            names=['timestamp_millsec', 'lin_acc_force_x_axis', 'lin_acc_force_y_axis',
                     'lin_acc_force_z_axis', 'accuracy', 'label'],
            dtype={'timestamp_millsec': np.uint64,
                    'lin_acc_force_x_axis': np.float32,
                    'lin_acc_force_y_axis': np.float32,
                    'lin_acc_force_z_axis': np.float32,
                    'accuracy': np.int8,
                    'label': 'category'})

#Convert milliseconds to seconds for our window evaluation
linear_accel_data['timestamp_sec'] = linear_accel_data['timestamp_millsec'].map(get_window)
linear_accel_data['resultant_lin_acc'] = np.sqrt(np.power(linear_accel_data['lin_acc_force_x_axis'],2) + np.power(linear_accel_data['lin_acc_force_y_axis'],2) + np.power(linear_accel_data['lin_acc_force_z_axis'],2))
#Cleans all values not in accuracy threshold
linear_accel_data = linear_accel_data[linear_accel_data.accuracy >= ACCURACY_THRESH]

#Get our acceleration features together
print("Getting acceleration features...")
acc_info = pd.DataFrame()
acc_info['result_acc_mean'] = accel_data.groupby('timestamp_sec')['resultant_acc'].mean()
acc_info['result_acc_median'] = accel_data.groupby('timestamp_sec')['resultant_acc'].median()
acc_info['result_acc_std'] = accel_data.groupby('timestamp_sec')['resultant_acc'].std()
acc_info['result_lin_acc_mean'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].mean()
acc_info['result_lin_acc_median'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].median()
acc_info['result_lin_acc_std'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].std()
acc_info.reset_index(level=acc_info.index.names, inplace=True)

#encode labels to categories
le = preprocessing.LabelEncoder()
le.fit(accel_data['label'])
print("Classes:", list(le.classes_))
accel_data['cat_label'] = le.transform(accel_data['label'])

#Get our labels for each time window
label = accel_data[['timestamp_sec','cat_label']]
label = label.drop_duplicates(subset=['timestamp_sec'], keep='first')

#make our training dataframe
full_info = pd.merge(acc_info, label, on='timestamp_sec')
num_rows = len(full_info)
div = int(num_rows / TEST_SPLIT)
split_pos = num_rows - div

train = full_info[:split_pos]
test = full_info[split_pos:]

#Write training features to file
train.to_csv(PATH + '/Data/Processed/train_features.csv', index=False, mode='w')
test.to_csv(PATH + '/Data/Processed/test_features.csv', index=False, mode='w')
print("Feature extraction completed in ", round(time.time() - start,4), "seconds")
