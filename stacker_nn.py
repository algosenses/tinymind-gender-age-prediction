import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split # to balance out label
import glob, os
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import LabelEncoder

output_location = 'Output/'
train_location = output_location + 'stacking_train_10fold/'
test_location = output_location + 'stacking_test_10fold/'

datadir = './data/Demo'
gatrain = pd.read_csv(os.path.join(datadir, 'deviceid_train.tsv'), delimiter='\t',
                      names=['device_id', 'gender', 'group'], index_col='device_id')
gatrain['label'] = gatrain['gender'].map(str) + '-' + gatrain['group'].map(str)

gatest = pd.read_csv(os.path.join(datadir, 'deviceid_test.tsv'), delimiter='\t',
                     names=['device_id'], index_col='device_id')

targetencoder = LabelEncoder().fit(gatrain.label)
y = targetencoder.transform(gatrain.label)
nclasses = len(targetencoder.classes_)


def build_model(input_size):
    model = Sequential()
    model.add(Dense(48, input_dim=input_size, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(24, input_dim=input_size, kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(Dense(nclasses, kernel_initializer='glorot_uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model


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


    skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True)

    y_dummy = to_categorical(y.tolist())

    train_predict_y = np.zeros((len(y), nclasses))
    test_predict_y = np.zeros((test_stacked.shape[0], nclasses))

    test_predict_list = []
    i = 0
    for train_idx, val_idx in skf.split(train_stacked, y):
        """ Each fold iteration """
        print('------------- fold round %d ------------' % i)
        """ Each fold cross validation """
        model = build_model(features)


        model.fit(train_stacked[train_idx], y_dummy[train_idx], batch_size=128,
                  epochs=40,
                  validation_data=(train_stacked[val_idx], y_dummy[val_idx]),
                  verbose=1
                  )


        scoring = model.predict_proba(train_stacked[val_idx])
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        print('    Fold %d score: %f' % (i, l_score))
        """test stack """
        tresult = model.predict_proba(test_stacked)
        test_predict_y = test_predict_y + tresult
        i += 1

    print('train prediction...')
    l_score = log_loss(y, train_predict_y)
    print('Final Fold score: %f' % (l_score))

    print('test prediction...')
    test_predict_y = test_predict_y / number_of_folds

    filename = 'neural_networ_stacked_'
    np.save(output_location + filename + 'test', test_predict_y)

    pred = pd.DataFrame(test_predict_y, index=gatest.index, columns=targetencoder.classes_)
    pred.index.name = 'DeviceID'
    cols = ['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10',
            '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']
    pred.to_csv('submission_stacker_nn.csv', index=True, columns=cols)

main()
