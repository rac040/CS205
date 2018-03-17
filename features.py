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
import scipy
import scipy.fftpack
from scipy import signal

#Run sensor data compilation
import compile_data

def get_dom_freq(samples, timestep, threshold):
    FFT = abs(scipy.fft(samples))
    freqs = scipy.fftpack.fftfreq(samples.size, timestep)
    idxs = np.where(np.logical_and(freqs>=threshold, freqs<=15))
    FFT = FFT[idxs]
    if FFT.size == 0:
        return float('NaN')
    freqs = freqs[idxs]
    max_sig = np.where(FFT == max(FFT))
    return freqs[max_sig][0]

def get_freq_features(samples, timestep):
    freqs, psd = signal.welch(samples, 1 / timestep)
#    FFT_orig = abs(scipy.fft(samples))
#    freqs = scipy.fftpack.fftfreq(samples.size, timestep)
    idxs = np.where(np.logical_and(freqs>=0.3, freqs<=15))
    psd_temp = psd[idxs]
    if psd_temp.size <= 1:
        return (float('NaN'),) * 8
    
#    if FFT.size <= 1:
#        return [float('NaN')] * 8
    freqs_temp = freqs[idxs]
    max_sig_idx = psd_temp.argmax()

    first_dom_freq = freqs_temp[max_sig_idx]
    if max_sig_idx == 0:
        freq_step = freqs_temp[1] - freqs_temp[0]
    else:
        freq_step = freqs_temp[max_sig_idx] - freqs_temp[max_sig_idx - 1]
    
    first_dom_pow = max(psd_temp) * freq_step
#    psd = np.square(FFT)
    tot_pow = np.trapz(psd_temp, freqs_temp)

    #second dom freq
    psd_temp[max_sig_idx] = 0
    max_sig_idx = psd_temp.argmax()

    if max_sig_idx == 0:
        freq_step = freqs_temp[1] - freqs_temp[0]
    else:
        freq_step = freqs_temp[max_sig_idx] - freqs_temp[max_sig_idx - 1]
    snd_dom_freq = freqs_temp[max_sig_idx]
    snd_dom_pow = max(psd_temp) * freq_step


    #third dom freq
    idxs = np.where(np.logical_and(freqs>=0.6, freqs<=2.5))
    psd_temp = psd[idxs]
    if psd_temp.size <= 1:
        return (float('NaN'),) * 8
#    if FFT.size == 0:
#        return [float('NaN')] * 8
    freqs_temp = freqs[idxs]
    max_sig_idx = psd_temp.argmax()

    if max_sig_idx == 0:
        freq_step = freqs_temp[1] - freqs_temp[0]
    else:
        freq_step = freqs_temp[max_sig_idx] - freqs_temp[max_sig_idx - 1]
    third_dom_freq = freqs_temp[max_sig_idx]
    third_dom_pow = max(psd_temp) * freq_step
    ratio = first_dom_pow / tot_pow

    return tot_pow, first_dom_freq, first_dom_pow, snd_dom_freq, snd_dom_pow, third_dom_freq, third_dom_pow, ratio

#Window size in seconds
WINDOW = 4
ACCURACY_THRESH = 2
TEST_SPLIT = 10
DATA_DIR = PATH + '/Data/Compiled/'
TEST_DATA_DIR = PATH + '/Data/Test/'
OUTFILE = PATH + '/Data/Processed/train_features.csv'
TEST_OUTFILE = PATH + '/Data/Processed/test_features.csv'
start = time.time()
print("\nStarting feature extraction...")
dirs = [DATA_DIR, TEST_DATA_DIR]
outfiles = [OUTFILE, TEST_OUTFILE]
for (idx, dir) in enumerate(dirs):
    outfile = outfiles[idx]
    #Get acceleration data from file
    accel_data = pd.read_csv(dir + '1_android.sensor.accelerometer.data.csv',
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
    linear_accel_data = pd.read_csv(dir + '10_android.sensor.linear_acceleration.data.csv',
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


    #Get frequency-domain info
    print("\tGetting frequency domain info...")

    acc_info['lin_timestep'] = linear_accel_data.groupby('timestamp_sec')['timestamp_millsec'].apply(lambda x: WINDOW / x.count())
    acc_info['timestep'] = accel_data.groupby('timestamp_sec')['timestamp_millsec'].apply(lambda x: WINDOW / x.count())

    linear_accel_data = pd.merge(linear_accel_data, acc_info, left_on = 'timestamp_sec', right_index=True)
    linear_accel_data['resultant_lin_acc_no_mean'] = linear_accel_data['resultant_lin_acc'] - linear_accel_data['result_lin_acc_mean']
    linear_accel_data = linear_accel_data.dropna()

    acc_freq_lin_info = linear_accel_data.groupby('timestamp_sec')[['resultant_lin_acc_no_mean', 'lin_timestep']].apply(lambda x: get_freq_features(x['resultant_lin_acc_no_mean'], timestep = x['lin_timestep'].iloc[0]))
    acc_freq_lin_info = acc_freq_lin_info.apply(pd.Series)
    acc_freq_lin_info.columns = ['tot_lin_pow', 'fst_dom_lin_freq', 'fst_dom_lin_pow', 'snd_dom_lin_freq', 'snd_dom_lin_pow', 'third_dom_lin_freq', 'third_dom_lin_pow', 'ratio_lin']
    acc_info = pd.merge(acc_info, acc_freq_lin_info, left_index=True, right_index=True)

    accel_data = pd.merge(accel_data, acc_info, left_on = 'timestamp_sec', right_index=True)
    accel_data['resultant_acc_no_mean'] = accel_data['resultant_acc'] - accel_data['result_acc_mean']
    accel_data = accel_data.dropna()
    acc_freq_info = accel_data.groupby('timestamp_sec')[['resultant_acc_no_mean', 'timestep']].apply(lambda x: get_freq_features(x['resultant_acc_no_mean'], timestep = x['timestep'].iloc[0]))

    acc_freq_info = acc_freq_info.apply(pd.Series)
    acc_freq_info.columns = ['tot_pow', 'fst_dom_freq', 'fst_dom_pow', 'snd_dom_freq', 'snd_dom_pow', 'third_dom_freq', 'third_dom_pow', 'ratio']
    acc_info = pd.merge(acc_info, acc_freq_info, left_index=True, right_index = True)

    acc_info = acc_info.dropna()

    acc_info.reset_index(level=acc_info.index.names, inplace=True)


    #Get orientation data from file
    print("Getting orientation features...")
    orient_data = pd.read_csv(dir + '3_android.sensor.orientation.data.csv',
                names=['timestamp_millsec', 'orient_north_y', 'orient_rot_x',
                         'oreint_rot_y', 'accuracy', 'label'],
                dtype={'timestamp_millsec': np.uint64,
                        'orient_north_y': np.float32,
                        'orient_rot_x': np.float32,
                        'oreint_rot_y': np.float32,
                        'accuracy': np.int8,
                        'label': 'category'})

    orient_data = orient_data[orient_data.accuracy >= ACCURACY_THRESH]
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
    step_data = pd.read_csv(dir + '19_android.sensor.step_counter.data.csv',
                names=['timestamp_millsec', 'steps', 'accuracy', 'label'],
                dtype={'timestamp_millsec': np.uint64,
                        'steps': np.uint64,
                        'accuracy': np.int8,
                        'label': 'category'})

    step_data = step_data[step_data.accuracy >= ACCURACY_THRESH]
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
    label = accel_data[['timestamp_sec','cat_label', 'label']]
    label = label.drop_duplicates(subset=['timestamp_sec'], keep='first')

    #make our training dataframe
    full_info = pd.merge(acc_info, label, on='timestamp_sec')
    full_info = pd.merge(full_info, orient_info, on='timestamp_sec')

    step_info = pd.merge(label, step_info, on='timestamp_sec', how='outer')
    full_info = pd.merge(full_info, step_info, on=['timestamp_sec', 'cat_label'], how='left')

    #Write training features to file
    full_info.to_csv(outfile, index=False, mode='w')
print("Feature extraction completed in ", round(time.time() - start,4), "seconds")
