import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import glob, os, sys
import time

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping

import xgboost as xgb

from first_level_model_list import model_list
from model_wrapper import BaseWrapper
from feature_eng import load_features, get_feature_code

nb_class = 0
Xtrain = []
Xtest = []
y = []
train_location = ''
test_location = ''

################################################################################
features = ['Brand', 'Model', 'InstalledApps', 'InstalledAppsNumber', 'BagOfAppCategories_TFIDF', 'AvgTimeOfAppCategoriesUsage',
            'KindsOfAppCatUsagePerDay', 'NumOfProcessUsagePerDay', 'TimeDistOfUserActivities', 'MostActiveHour']

Xtrain, Xtest = load_features(features)

datadir = './data/Demo'
gatrain = pd.read_csv(os.path.join(datadir, 'deviceid_train.tsv'), delimiter='\t',
                      names=['device_id', 'gender', 'group'], index_col='device_id')
gatrain['label'] = gatrain['gender'].map(str) + '-' + gatrain['group'].map(str)

gatest = pd.read_csv(os.path.join(datadir, 'deviceid_test.tsv'), delimiter='\t',
                     names=['device_id'], index_col='device_id')

targetencoder = LabelEncoder().fit(gatrain.label)
y = targetencoder.transform(gatrain.label)
nb_class = len(targetencoder.classes_)

################################################################################
# Selection using chi-square
print("# Feature Selection")
selector = SelectKBest(chi2, k=8000).fit(Xtrain, y)
Xtrain = selector.transform(Xtrain)
Xtest = selector.transform(Xtest)


################################################################################
def model_MLP_builder():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain.shape[1], kernel_initializer='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=Xtest.shape[1], kernel_initializer='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(nb_class, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  # logloss

    return model


def learning_MLP(number_of_folds=10, seed=42):
    model_name = 'MLP'
    print('Model: %s' % model_name)

    """ Each model iteration """
    train_predict_y = np.zeros((len(y), nb_class))
    test_predict_y = np.zeros((Xtest.shape[0], nb_class))
    ll = 0.
    """ Important to set seed """
    skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=seed)
    """ Each fold cross validation """
    i = 0
    for train_idx, val_idx in skf.split(Xtrain, y):
        print('Fold ', i + 1)
        model = model_MLP_builder()
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(Xtrain[train_idx], y[train_idx], batch_size=128, epochs=20,
                  validation_data=(Xtrain[val_idx].todense(), y[val_idx]),
                  callbacks=[early_stopping])
        scoring = model.predict_proba(Xtrain[val_idx])
        """ Out of fold prediction """
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        ll += l_score
        print('    Fold %d score: %f' % (i + 1, l_score))
        i += 1

    print('average val log_loss: %f' % (ll / number_of_folds))
    """ Fit Whole Data and predict """
    print('training whole data for test prediction...')
    model = model_MLP_builder()
    model.fit(Xtrain, y, batch_size=128, epochs=20)
    test_predict_y = model.predict_proba(Xtest)

    filename = model_name + '_' + str(number_of_folds) + 'fold'
    np.save(train_location + filename + '_train', train_predict_y)
    np.save(test_location + filename + '_test', test_predict_y)


################################################################################
def model_XGB_builder():
    parameters = {
        'eta': 0.01,
        'max_depth': 6,
        'num_boost_round': 100,
        'alpha': 2,
        'colsample_bytree': 1.0,
        'lambda': 3,
        'subsample': 1.0
    }

    model = xgb.XGBClassifier(param_grid=parameters, n_estimators=400, seed=42, objective='multi:softmax')
    return model


def learning_XGB(number_of_folds=10, seed=42):
    model_name = 'XGB'
    print('Model: %s' % model_name)

    """ Each model iteration """
    train_predict_y = np.zeros((len(y), nb_class))
    test_predict_y = np.zeros((Xtest.shape[0], nb_class))
    ll = 0.
    """ Important to set seed """
    skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=seed)
    """ Each fold cross validation """
    i = 0
    for train_idx, val_idx in skf.split(Xtrain, y):
        print('Fold ', i + 1)
        model = model_XGB_builder()
        model.fit(Xtrain[train_idx], y[train_idx],
                  early_stopping_rounds=35, eval_metric="mlogloss", eval_set=[(Xtrain[val_idx], y[val_idx])])
        scoring = model.predict_proba(Xtrain[val_idx])
        """ Out of fold prediction """
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        ll += l_score
        print('    Fold %d score: %f' % (i + 1, l_score))
        i += 1

    print('average val log_loss: %f' % (ll / number_of_folds))
    """ Fit Whole Data and predict """
    print('training whole data for test prediction...')
    model = model_XGB_builder()
    model.fit(Xtrain, y)
    test_predict_y = model.predict_proba(Xtest)

    filename = model_name + '_' + str(number_of_folds) + 'fold'
    np.save(train_location + filename + '_train', train_predict_y)
    np.save(test_location + filename + '_test', test_predict_y)


################################################################################
def model_LR_builder():
    model = LogisticRegression(C=0.1, multi_class='multinomial', penalty='l2', solver='newton-cg')
    return model

def learning_LR(number_of_folds=10, seed=42):
    model_name = 'LR'
    print('Model: %s' % model_name)

    """ Each model iteration """
    train_predict_y = np.zeros((len(y), nb_class))
    test_predict_y = np.zeros((Xtest.shape[0], nb_class))
    ll = 0.
    """ Important to set seed """
    skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=seed)
    """ Each fold cross validation """
    i = 0
    for train_idx, val_idx in skf.split(Xtrain, y):
        print('Fold ', i + 1)
        model = model_LR_builder()
        model.fit(Xtrain[train_idx], y[train_idx])
        scoring = model.predict_proba(Xtrain[val_idx])
        """ Out of fold prediction """
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        ll += l_score
        print('    Fold %d score: %f' % (i + 1, l_score))
        i += 1

    print('average val log_loss: %f' % (ll / number_of_folds))
    """ Fit Whole Data and predict """
    print('training whole data for test prediction...')
    model = model_LR_builder()
    model.fit(Xtrain, y)
    test_predict_y = model.predict_proba(Xtest)

    filename = model_name + '_' + str(number_of_folds) + 'fold'
    np.save(train_location + filename + '_train', train_predict_y)
    np.save(test_location + filename + '_test', test_predict_y)


if __name__ == "__main__":
    np.random.seed(42)
    train_location = "Output/stacking_train_10fold/"
    test_location = "Output/stacking_test_10fold/"

    print('prediction class: ', nb_class)
    print('seed: ', 42)
    print('train output: ', train_location)
    print('test output: ', test_location)

    print('train data features: %d' % (Xtrain.shape[1]))
    print('test data features: %d' % Xtest.shape[1])

    learning_MLP()
    learning_LR()
    learning_XGB()
