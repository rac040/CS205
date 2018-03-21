import os
import sys
PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DIR = PATH + '/Data/Results/'

IS_VALIDATION = False
PRINT_METRICS = True

start = time.time()

if IS_VALIDATION:
    orig_data = pd.read_csv(PATH + '/Data/Processed/train_features.csv',
                          dtype={'timestamp_millsec': np.uint64,
                            'cat_label': 'category'})

    train_x, test_x = train_test_split(orig_data, test_size=0.2)
    
    train_x.reset_index(level=train_x.index.names, inplace=True)
    test_x.reset_index(level=test_x.index.names, inplace=True)
else:
    print("GETTING TRAINING FEATURES...")
    train_x = pd.read_csv(PATH + '/Data/Processed/train_features.csv',
                          dtype={'timestamp_millsec': np.uint64,
                            'cat_label': 'category'})
    print("\tDone.")



    print("GETTING TESTING FEATURES...")
    test_x = pd.read_csv(PATH + '/Data/Processed/test_features.csv',
                          dtype={'timestamp_millsec': np.uint64,
                            'cat_label': 'category'})
    print("\tDone.")

print("GETTING TRAINING LABELS...")
train_labels = train_x['cat_label']
print("\tDone.")

features = ['timestamp_sec','result_acc_mean', 'result_acc_median',
            'result_acc_std', 'result_acc_max', 'result_acc_min',
            'result_acc_cross_median', 'result_lin_acc_mean',
            'result_lin_acc_median', 'result_lin_acc_std', 'result_lin_acc_max',
            'result_lin_acc_min','result_orient_mean', 'result_orient_median',
            'result_orient_std','ratio_lin']

##'lin_timestep', 'timestep','tot_lin_pow', 'fst_dom_lin_freq', 'fst_dom_lin_pow',
##'snd_dom_lin_freq', 'snd_dom_lin_pow', 'third_dom_lin_freq', 'third_dom_lin_pow',
##'ratio_lin', 'tot_pow', 'fst_dom_freq', 'fst_dom_pow', 'snd_dom_freq', 'snd_dom_pow',
##'third_dom_freq', 'third_dom_pow', 'ratio'

#'result_acc_mean', 'result_acc_median',
#            'result_acc_std', 'result_acc_max', 'result_acc_min',
#            'result_acc_cross_median', 'result_lin_acc_mean',
#            'result_lin_acc_median', 'result_lin_acc_std', 'result_lin_acc_max',
#            'result_lin_acc_min','result_orient_mean', 'result_orient_median',
#            'result_orient_std','steps_mean'
#steps_std doesn't seem to help, but steps_mean does slightly

# parameter for lgbt#0.38119
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',

    'learning_rate': 0.004953,
    'max_depth': 6,
    'num_leaves': 50,
    'min_data_in_leaf': 10,
    
    'feature_fraction': 0.2,
    'bagging_fraction': 0.55,
    'bagging_freq': 15,
    'max_bin': 5
}
num_round = 500

print('Building dataset...')
# keep features
train_data = lgb.Dataset(train_x[features], label=train_labels)
valid_data = lgb.Dataset(train_x[features], train_labels)

# starting to train
print('Training......')
evals_result = {}
bst = lgb.train(params, train_data, num_round,
                valid_sets=valid_data, verbose_eval=5,evals_result=evals_result)
del train_x

if PRINT_METRICS:
    print('Plot metrics during training...')
    ax = lgb.plot_metric(evals_result, metric='multi_logloss')
    plt.show()

    print('Plot feature importances...')
    ax = lgb.plot_importance(bst, max_num_features=len(features))
    plt.show()

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
print(str(round(((end - start) / 60), 6)), "minutes")
