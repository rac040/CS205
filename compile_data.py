import os
import sys
import csv
import glob
import gzip

PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = PATH + '/Data/Sessions/'
TEST_DATA_DIR = PATH + '/Data/test_data/'
OUTPUT_DIR = PATH + '/Data/Compiled/Train/'
TEST_DIR = PATH + '/Data/Compiled/Test/'
#\Data\Sessions\14442D38F8ACC8E_Fri_Mar_09_09-21_2018_PST\data

print("Removing old compiled files...")
files = glob.glob(OUTPUT_DIR + '*.csv')
for f in files:
    os.remove(f)

files = glob.glob(TEST_DIR + '*.csv')
for f in files:
    os.remove(f)

print("Compiling sensor data...")
for subdir, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".gz"):
            filename = os.path.join(subdir, file)
            #print(filename)
            sensor_name = (filename.split('T/data/')[1]).split(".gz")[0]
            #print(sensor_name)

            with gzip.open(filename, 'r') as f:
                try:
                    file_content = f.read()
                    wfile = open(OUTPUT_DIR + sensor_name, "ab")
                    wfile.write(file_content)
                    wfile.close()
                    
                except EOFError:
                    print("ERROR READING:", filename)
                    pass

                f.close()

print("Compiling test data...")
for subdir, dirs, files in os.walk(TEST_DATA_DIR):
    for file in files:
        if file.endswith(".gz"):
            filename = os.path.join(subdir, file)
            #print(filename)
            sensor_name = (filename.split('T/data/')[1]).split(".gz")[0]
            #print(sensor_name)
            
            with gzip.open(filename, 'r') as f:
                try:
                    file_content = f.read()
                    wfile = open(TEST_DIR + sensor_name, "ab")
                    wfile.write(file_content)
                    wfile.close()
                
                except EOFError:
                    print("ERROR READING:", filename)
                    pass
                
                f.close()
