import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

output_location = 'Output/'

train_location = output_location + 'first_level_train/'
test_location = output_location + 'second_level_collections/'

datadir = './data/Demo'
gatrain = pd.read_csv(os.path.join(datadir, 'deviceid_train.tsv'), delimiter='\t',
                      names=['device_id', 'gender', 'group'], index_col='device_id')
gatrain['label'] = gatrain['gender'].map(str) + '-' + gatrain['group'].map(str)

gatest = pd.read_csv(os.path.join(datadir, 'deviceid_test.tsv'), delimiter='\t',
                     names=['device_id'], index_col='device_id')

targetencoder = LabelEncoder().fit(gatrain.label)
y = targetencoder.transform(gatrain.label)
nclasses = len(targetencoder.classes_)

ensemble_name = 'ensembled_'
train_size = 50000
test_size = 22727


size = test_size
directory = test_location

train_data = False
if train_data:
    size = train_size
    directory = train_location

files = []
total_file = 0
print('------------- Ensemble ------------')
for file in os.listdir(directory):
    if file.endswith(".npy"):
        print('file number[%d]: %s' % (total_file, file))
        temp = np.load(directory + file)
        files.append((file, temp))
        total_file += 1


ensembles = []
weights = []

for f in files:
    ensembles.append(f)

weights = [0.7, 0.3]

final_result = np.zeros((size, 22))
for i in range(len(ensembles)):
    print(' [%d] %s' % (i, ensembles[i][0]))
    final_result += ensembles[i][1] * weights[i]


pred = pd.DataFrame(final_result, index=gatest.index, columns=targetencoder.classes_)
pred.index.name = 'DeviceID'
cols = ['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10',
        '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']
pred.to_csv('submission_ensemble_weighted.csv', index=True, columns=cols)
