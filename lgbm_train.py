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
            'result_acc_std', 'result_lin_acc_mean', 'result_lin_acc_median',
            'result_lin_acc_std']

#'comp_size', 'avg_diff', 'std_diff'
#'orders_sum', 'days_since_prior_std','avg_basket', 'avg_reorder', 'num_unique_items',
#'comp_size', 'avg_diff', 'std_diff' 'order_ratio',
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
pred = pd.DataFrame(bst.predict(test_x[features]))
pred.columns = ['laying', 'null','sitting','standing','walking']
print('Prediction Done......')
results = test_x.join(pred)
results[['timestamp_sec', 'laying', 'null','sitting','standing','walking']].to_csv(DIR + 'lgbm_pred.csv', mode='w+', index=False)
end = time.time()
print(str((end - start) / 60), "minutes")
