import os
import sys
PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import itertools

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

DIR = PATH + '/Data/Results/'

IS_VALIDATION = False
PRINT_METRICS = False

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
    ax = lgb.plot_importance(bst, max_num_features=len(features), importance_type = 'split')
    plt.show()

print('Predicting......')
results = pd.DataFrame(bst.predict(test_x[features]))
#pred.columns = ['laying', 'null','sitting','standing','walking']
results['pred_label'] = results.idxmax(axis=1)
results['actual_label'] = test_x['cat_label']
results['timestamp_sec'] = test_x['timestamp_sec']
results.to_csv(DIR + 'lgbm_pred.csv', mode='w+', index=False)
print('Prediction Done......')

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#Get a score for our results
check_frame = results[['pred_label', 'actual_label']]
num_predictions = len(check_frame)
num_correct = 0

cnf_matrix = confusion_matrix(results['actual_label'].astype(int),
                              results['pred_label'].astype(int))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['laying', 'sitting',
                                           'standing','walking'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['laying', 'sitting'
                                           ,'standing','walking'],
                      normalize=True,
                      title='Normalized confusion matrix')

for row in check_frame.itertuples(index=False, name=None):
    if int(row[0]) == int(row[1]):
        num_correct = num_correct + 1

print("\nPercent of Predictions Correct:", (num_correct / num_predictions) * 100)

end = time.time()
print(str(round(((end - start) / 60), 6)), "minutes")
