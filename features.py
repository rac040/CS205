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

def get_features(isTrain = True):
    if isTrain:
        DATA_DIR = PATH + '/Data/Compiled/Train/'
    else:
        DATA_DIR = PATH + '/Data/Compiled/Test/'

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
    accel_data = accel_data[accel_data.label != 'null']

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
    linear_accel_data = linear_accel_data[linear_accel_data.label != 'null']

    #Get our acceleration features together
    print("Getting acceleration features...")
    acc_info = pd.DataFrame()
    acc_info['result_acc_mean'] = accel_data.groupby('timestamp_sec')['resultant_acc'].mean()
    acc_info['result_acc_median'] = accel_data.groupby('timestamp_sec')['resultant_acc'].median()
    acc_info['result_acc_std'] = accel_data.groupby('timestamp_sec')['resultant_acc'].std()
    acc_info['result_acc_max'] = accel_data.groupby('timestamp_sec')['resultant_acc'].max()
    acc_info['result_acc_min'] = accel_data.groupby('timestamp_sec')['resultant_acc'].min()

    def cross_med(df):
        median = df.median()
        df = pd.DataFrame(df)

        num_cross = 0
        prev_val = None
        for row in df.itertuples(index=False, name=None):
            cur_val = row[0]
            if prev_val != None:
                if (cur_val > median and prev_val < median) or (cur_val < median and prev_val > median):
                    num_cross = num_cross + 1

            prev_val = cur_val

        return num_cross
    print("\tGetting accel_median crossings...")
    acc_info['result_acc_cross_median'] = accel_data.groupby('timestamp_sec')['resultant_acc'].apply(cross_med)

    acc_info['result_lin_acc_mean'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].mean()
    acc_info['result_lin_acc_median'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].median()
    acc_info['result_lin_acc_std'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].std()
    acc_info['result_lin_acc_max'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].max()
    acc_info['result_lin_acc_min'] = linear_accel_data.groupby('timestamp_sec')['resultant_lin_acc'].min()
    acc_info.reset_index(level=acc_info.index.names, inplace=True)


    #Get orientation data from file
    print("Getting orientation features...")
    orient_data = pd.read_csv(DATA_DIR + '3_android.sensor.orientation.data.csv',
                names=['timestamp_millsec', 'orient_north_y', 'orient_rot_x',
                         'oreint_rot_y', 'accuracy', 'label'],
                dtype={'timestamp_millsec': np.uint64,
                        'orient_north_y': np.float32,
                        'orient_rot_x': np.float32,
                        'oreint_rot_y': np.float32,
                        'accuracy': np.int8,
                        'label': 'category'})

    orient_data = orient_data[orient_data.accuracy >= ACCURACY_THRESH]
    orient_data = orient_data[orient_data.label != 'null']
    
    orient_data['timestamp_sec'] = orient_data['timestamp_millsec'].map(get_window)
    orient_data['resultant_orient'] = np.sqrt(np.power(orient_data['orient_north_y'],
                                                    2) + np.power(orient_data['orient_rot_x'],
                                                    2) + np.power(orient_data['oreint_rot_y'],
                                                    2))
    #Orientation features
    orient_info = pd.DataFrame()
    orient_info['result_orient_mean'] = orient_data.groupby('timestamp_sec')['resultant_orient'].mean()
    orient_info['result_orient_median'] = orient_data.groupby('timestamp_sec')['resultant_orient'].median()
    orient_info['result_orient_std'] = orient_data.groupby('timestamp_sec')['resultant_orient'].std()
    orient_info.reset_index(level=orient_info.index.names, inplace=True)

    #Get step counts from file
    print("Getting step count features...")
    #19_android.sensor.step_counter.data.csv
    step_data = pd.read_csv(DATA_DIR + '19_android.sensor.step_counter.data.csv',
                names=['timestamp_millsec', 'steps', 'accuracy', 'label'],
                dtype={'timestamp_millsec': np.uint64,
                        'steps': np.uint64,
                        'accuracy': np.int8,
                        'label': 'category'})

    step_data = step_data[step_data.accuracy >= ACCURACY_THRESH]
    step_data = step_data[step_data.label != 'null']
    step_data['timestamp_sec'] = step_data['timestamp_millsec'].map(get_window)

    #step count features
    step_info = pd.DataFrame()
    step_info['steps_mean'] = step_data.groupby('timestamp_sec')['steps'].mean()
    step_info['steps_std'] = step_data.groupby('timestamp_sec')['steps'].std()
    step_info.reset_index(level=step_info.index.names, inplace=True)

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
    full_info = pd.merge(full_info, orient_info, on='timestamp_sec')

    step_info = pd.merge(label, step_info, on='timestamp_sec', how='outer')
    full_info = pd.merge(full_info, step_info, on=['timestamp_sec', 'cat_label'], how='left')

    num_rows = len(full_info)
    div = int(num_rows / TEST_SPLIT)
    split_pos = num_rows - div

    #Write training features to file
    if isTrain:
        full_info.to_csv(PATH + '/Data/Processed/train_features.csv', index=False, mode='w')
    else: 
        full_info.to_csv(PATH + '/Data/Processed/test_features.csv', index=False, mode='w')
    print("Feature extraction completed in ", round(time.time() - start,4), "seconds")

print("Getting Training Features...")
get_features(True)
print("Getting Testing Features...")
get_features(False)
