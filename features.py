import os
import sys
PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import lightgbm as lgb
import csv
import gc
import time

SESSION = 1
DATA_DIR = PATH + '/Data/Session ' + str(SESSION) + '/data/'

start = time.time()

#Get acceleration data from file
accel_data = pd.read_csv(DATA_DIR + '1_android.sensor.accelerometer.data.csv',
            names=['timestamp_millsec', 'acc_force_x_axis', 'acc_force_y_axis',
                     'acc_force_z_axis', 'accuracy', 'label'],
            dtype={'timestamp_millsec': np.uint64,
                    'acc_force_x_axis': np.float32,
                    'acc_force_y_axis': np.float32,
                    'acc_force_z_axis': np.float32,
                    'accuracy': np.int8,
                    'label': np.str})

#Convert milliseconds to seconds for our window evaluation
accel_data['timestamp_sec'] = accel_data['timestamp_millsec'].map(lambda x: int(float(x) * float(pow(10,-3))))
accel_data['resultant_acc'] = np.sqrt(np.power(accel_data['acc_force_x_axis'],2) + np.power(accel_data['acc_force_y_axis'],2) + np.power(accel_data['acc_force_z_axis'],2))

#Get acceleration data from file
linear_accel_data = pd.read_csv(DATA_DIR + '10_android.sensor.linear_acceleration.data.csv',
            names=['timestamp_millsec', 'lin_acc_force_x_axis', 'lin_acc_force_y_axis',
                     'lin_acc_force_z_axis', 'accuracy', 'label'],
            dtype={'timestamp_millsec': np.uint64,
                    'lin_acc_force_x_axis': np.float32,
                    'lin_acc_force_y_axis': np.float32,
                    'lin_acc_force_z_axis': np.float32,
                    'accuracy': np.int8,
                    'label': np.str})

#Convert milliseconds to seconds for our window evaluation
linear_accel_data['timestamp_sec'] = linear_accel_data['timestamp_millsec'].map(lambda x: int(float(x) * float(pow(10,-3))))
linear_accel_data['resultant_lin_acc'] = np.sqrt(np.power(linear_accel_data['lin_acc_force_x_axis'],2) + np.power(linear_accel_data['lin_acc_force_y_axis'],2) + np.power(linear_accel_data['lin_acc_force_z_axis'],2))

#Get our acceleration features together
acc_info = pd.DataFrame()
acc_info['result_acc_mean'] = accel_data.groupby('timestamp_sec')['resultant_acc'].mean()
acc_info['result_acc_std'] = accel_data.groupby('timestamp_sec')['resultant_acc'].std()
acc_info['result_lin_acc_mean'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].mean()
acc_info['result_lin_acc_std'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].std()
acc_info.reset_index(level=acc_info.index.names, inplace=True)

#Get our labels for each time window
label = accel_data[['timestamp_sec','label']]
label = label.drop_duplicates(subset=['timestamp_sec'], keep='first')

#make our training dataframe
train = pd.merge(acc_info, label, on='timestamp_sec')






