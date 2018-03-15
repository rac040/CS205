import os
import sys
PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from datetime import datetime, timedelta
DIR = PATH + '/Data/Results/'

start = time.time()

print("GETTING TRAINING FEATURES...")
train_x = pd.read_csv(PATH + '/Data/Processed/train_features.csv',
                      dtype={'timestamp_millsec': np.uint64,
                        'cat_label': 'category'})
print("\tDone.")

print("GETTING TRAINING LABELS...")
train_labels = train_x['cat_label']
print("\tDone.")

print("GETTING TESTING FEATURES...")
test_x = pd.read_csv(PATH + '/Data/Processed/test_features.csv',
                      dtype={'timestamp_millsec': np.uint64,
                        'cat_label': 'category'})
print("\tDone.")

features = ['timestamp_sec', 'result_acc_mean', 'result_acc_median',
            'result_acc_std', 'result_acc_max', 'result_acc_min',
            'result_acc_cross_median', 'result_lin_acc_mean',
            'result_lin_acc_median', 'result_lin_acc_std', 'result_lin_acc_max',
            'result_lin_acc_min','result_orient_mean', 'result_orient_median',
            'result_orient_std','steps_mean']
#steps_std doesn't seem to help, but steps_mean does slightly

# parameter for lgbt#0.38119
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 5,
    'metric': {'multi_logloss'},
    'max_depth': 5,
    'learning_rate': 0.05,
    'num_leaves': 32,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
num_round = 120

print('Building dataset...')
# keep features
train_data = lgb.Dataset(train_x[features], label=train_labels)
valid_data = lgb.Dataset(train_x[features], train_labels)

# starting to train
print('Training......')
bst = lgb.train(params, train_data, num_round, valid_sets=valid_data, verbose_eval=5)
del train_x

print('Predicting......')
results = pd.DataFrame(bst.predict(test_x[features]))
#pred.columns = ['laying', 'null','sitting','standing','walking']
results['pred_label'] = results.idxmax(axis=1)
results['actual_label'] = test_x['cat_label']
results['timestamp_sec'] = test_x['timestamp_sec']
results.to_csv(DIR + 'lgbm_pred.csv', mode='w+', index=False)
print('Prediction Done......')

#Get a score for our results
check_frame = results[['pred_label', 'actual_label']]
num_predictions = len(check_frame)
num_correct = 0

for row in check_frame.itertuples(index=False, name=None):
    if int(row[0]) == int(row[1]):
        num_correct = num_correct + 1

print("\nPercent of Predictions Correct:", (num_correct / num_predictions) * 100)

end = time.time()
print(str((end - start) / 60), "minutes")
