import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split # to balance out label
import glob, os
import time
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

output_location = 'Output/'
train_location = output_location + 'stacking_train_10fold/'
test_location = output_location + 'stacking_test_10fold/'
stacking_area = output_location + 'bagging/'

datadir = './data/Demo'
gatrain = pd.read_csv(os.path.join(datadir, 'deviceid_train.tsv'), delimiter='\t',
                      names=['device_id', 'gender', 'group'], index_col='device_id')
gatrain['label'] = gatrain['gender'].map(str) + '-' + gatrain['group'].map(str)

gatest = pd.read_csv(os.path.join(datadir, 'deviceid_test.tsv'), delimiter='\t',
                     names=['device_id'], index_col='device_id')

targetencoder = LabelEncoder().fit(gatrain.label)
y = targetencoder.transform(gatrain.label)
nclasses = len(targetencoder.classes_)


def main():
    """ Prepare stack data. load every 1st learner's predictions and stack them together """
    stack = []
    for file in os.listdir(train_location):
        if file.endswith(".npy"):
            temp = np.load(train_location + file)
            stack.append(temp)
            print('file: %s' % file)
    print('total: %d ' % len(stack))

    test_stack = []
    for file in os.listdir(test_location):
            if file.endswith(".npy"):
                temp = np.load(test_location + file)
                test_stack.append(temp)
                print('file: %s' % file)
    print('total: %d ' % len(test_stack))


    train_stacked = np.hstack(stack)
    test_stacked = np.hstack(test_stack)
    features = train_stacked.shape[1]

    del stack, test_stack
    number_of_folds = 5
    number_of_bagging = 1


    d_test = xgb.DMatrix(test_stacked)


    bagging = 0
    bag_of_predictions = np.zeros((test_stacked.shape[0], nclasses))
    for x in range(number_of_bagging):

        """ Each bagging iteration """
        print('------------- bagging round %d ------------' % (x + 1))
        test_predict_y = np.zeros((test_stacked.shape[0], nclasses))

        """ Important to set seed """
        skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=None)

        i = 0
        for train_idx, val_idx in skf.split(train_stacked, y):
            print('Fold ', i + 1)
            """ Create DMatrix """
            d_train = xgb.DMatrix(train_stacked[train_idx], label=y[train_idx])
            d_val = xgb.DMatrix(train_stacked[val_idx], label=y[val_idx])
            """ Each fold cross validation """
            param = {}
            # use softmax multi-class classification
            param['objective'] = 'multi:softprob'
            # scale weight of positive examples
            param['eta'] = 0.01
            param['max_depth'] = 9
            param['num_class'] = nclasses
            param['silent'] = 1
            param['alpha'] = 3
            param['eval_metric'] = 'mlogloss'

            watchlist = [(d_train, 'train'), (d_val, 'validation')]
            num_round = 1000
            bst = xgb.train(param, d_train, num_round, watchlist, early_stopping_rounds=100, verbose_eval=50)

            """ Cal test_predict_y """
            t_scores = bst.predict(d_test)
            test_predict_y = test_predict_y + t_scores

            i += 1

        """ CV result """
        test_predict_y = test_predict_y / number_of_folds
        bag_of_predictions = bag_of_predictions + test_predict_y


    bag_of_predictions = bag_of_predictions / number_of_bagging
    filename = 'xgb_stacker_'
    np.save(stacking_area + filename + '10fold_alpha1', bag_of_predictions)

    pred = pd.DataFrame(test_predict_y, index=gatest.index, columns=targetencoder.classes_)
    pred.index.name = 'DeviceID'
    cols = ['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10',
            '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']
    pred.to_csv('submission_stacker_xgb.csv', index=True, columns=cols)

main()
