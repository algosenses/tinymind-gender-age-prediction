from feature_base import BaseFeature
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix, hstack

datadir = './data/Demo'
gatrain = pd.read_csv(os.path.join(datadir, 'deviceid_train.tsv'), delimiter='\t',
                      names=['device_id', 'gender', 'group'], index_col='device_id')
gatrain['label'] = gatrain['gender'].map(str) + '-' + gatrain['group'].map(str)

gatest = pd.read_csv(os.path.join(datadir, 'deviceid_test.tsv'), delimiter='\t',
                     names=['device_id'], index_col='device_id')

phone = pd.read_csv(os.path.join(datadir, 'deviceid_brand.tsv'), delimiter='\t',
                            names=['device_id', 'phone_brand', 'device_model'])

phone = phone.drop_duplicates('device_id', keep='first').set_index('device_id')

device_packages = pd.read_csv(os.path.join(datadir, 'deviceid_packages.tsv'), delimiter='\t',
                              names=['device_id', 'packages'], index_col='device_id')

package_label = pd.read_csv(os.path.join(datadir, 'package_label.tsv'), delimiter='\t',
                            names=['package', 'class', 'subclass'], index_col='package')


class Brand(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'Brand'
        self.description = 'Mobile Phone Brand'

    def build(self):
        phone.phone_brand = phone.phone_brand.astype(str)
        brands = phone.phone_brand.unique()
        np.append(brands, 'unknown')
        brandencoder = LabelBinarizer().fit(brands)

        gatrain['brand'] = phone['phone_brand']
        gatrain['brand'].fillna('unknown', inplace=True)
        Xtr_brand = brandencoder.transform(gatrain[['brand']])
        Xtr_brand = csr_matrix(Xtr_brand)

        gatest['brand'] = phone['phone_brand']
        gatest['brand'].fillna('unknown', inplace=True)
        Xte_brand = brandencoder.transform(gatest['brand'])
        Xte_brand = csr_matrix(Xte_brand)

        return Xtr_brand, Xte_brand


class Model(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'Model'
        self.description = 'Mobile Phone Model'

    def build(self):
        phone['model'] = phone.phone_brand.str.cat(phone.device_model)
        models = phone['model'].unique()
        np.append(models, 'unknown')
        modelencoder = LabelBinarizer().fit(models.astype(str))

        gatrain['model'] = phone['model']
        gatrain['model'].fillna('unknown', inplace=True)
        Xtr_model = modelencoder.transform(gatrain[['model']])
        Xtr_model = csr_matrix(Xtr_model)

        gatest['model'] = phone['model']
        gatest['model'].fillna('unknown', inplace=True)
        Xte_model = modelencoder.transform(gatest['model'])
        Xte_model = csr_matrix(Xte_model)

        return Xtr_model, Xte_model


class InstalledApps(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'InstalledApps'
        self.description = 'Installed applications'

    def build(self):
        device_packages['apps'] = device_packages['packages'].apply(lambda x: x.split(','))
        appencoder = MultiLabelBinarizer().fit(device_packages['apps'])
        napps = len(appencoder.classes_)

        gatrain['apps'] = device_packages['apps']
        Xtr_app = appencoder.transform(gatrain['apps'])
        Xtr_app = csr_matrix(Xtr_app)

        gatest['apps'] = device_packages['apps']
        Xte_app = appencoder.transform(gatest['apps'])
        Xte_app = csr_matrix(Xte_app)

        return Xtr_app, Xte_app


class InstalledAppsNumber(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'InstalledAppsNumber'
        self.description = 'Number of installed applications'

    def build(self):
        device_packages['apps'] = device_packages['packages'].apply(lambda x: x.split(','))
        appencoder = MultiLabelBinarizer().fit(device_packages['apps'])
        napps = len(appencoder.classes_)

        gatrain['apps'] = device_packages['apps']
        Xtr_appnum = gatrain['apps'].apply(lambda x: len(x))
        Xtr_appnum = csr_matrix(Xtr_appnum).T

        gatest['apps'] = device_packages['apps']
        Xte_appnum = gatest['apps'].apply(lambda x: len(x))
        Xte_appnum = csr_matrix(Xte_appnum).T

        return Xtr_appnum, Xte_appnum


class BagOfAppCategories_TFIDF(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'BagOfAppCategories_TFIDF'
        self.description = 'IFIDF weighted of installed application categories'

    def build(self):
        package_label['category'] = package_label['class'].map(str) + '-' + package_label['subclass'].map(str)
        categories = package_label['category'].unique()
        np.append(categories, 'unknown')

        pkg2cat = {}
        for idx, row in package_label.iterrows():
            pkg2cat.update({idx: row['category']})

        def get_pkg_cat(pkg_list):
            cats = []
            for pkg in pkg_list:
                if pkg in pkg2cat:
                    cats.append(pkg2cat[pkg])
                else:
                    cats.append('unknown')

            return cats

        device_packages['packages'] = device_packages['packages'].apply(lambda s: s.split(','))
        device_packages['categories'] = device_packages['packages'].apply(get_pkg_cat)

        vectorizer = TfidfVectorizer(min_df=1)

        gatrain['categories'] = device_packages['categories'].apply(lambda x: ' '.join(s for s in x))
        gatrain['categories'].fillna('unknown', inplace=True)
        gatest['categories'] = device_packages['categories'].apply(lambda x: ' '.join(s for s in x))
        gatest['categories'].fillna('unknown', inplace=True)

        vectorizer.fit(pd.concat([gatrain['categories'], gatest['categories']], axis=0))

        Xtr_bagofcat = vectorizer.transform(gatrain['categories'])
        Xtr_bagofcat = csr_matrix(Xtr_bagofcat)

        Xte_bagofcat = vectorizer.transform(gatest['categories'])
        Xte_bagofcat = csr_matrix(Xte_bagofcat)

        return Xtr_bagofcat, Xte_bagofcat


class AvgTimeOfAppsUsage(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'AvgTimeOfAppsUsage'
        self.description = 'Average time of application usage'

    def build(self):
        Xtrain_data = pd.read_csv(os.path.join(self._save_path, 'Xtrain_TimeOfAppsUsage.tsv'), delimiter='\t',
                                           names=['row', 'col', 'total', 'avg'])

        Xtest_data = pd.read_csv(os.path.join(self._save_path, 'Xtest_TimeOfAppsUsage.tsv'), delimiter='\t',
                                          names=['row', 'col', 'total', 'avg'])

        napps = 35000
        onehot_encoding = False

        if onehot_encoding:
            # one-hot encoding
            Xtrain = np.ones((Xtrain_data['avg'].shape[0],), dtype=np.int)
            Xtest = np.ones((Xtest_data['avg'].shape[0],), dtype=np.int)
        else:
            # weighted encoding
            Xtrain = Xtrain_data['avg']
            Xtest = Xtest_data['avg']

        Xtr_avgtime = csr_matrix((Xtrain, (Xtrain_data['row'], Xtrain_data['col'])),
                                  shape=(gatrain.shape[0], napps))
        Xte_avgtime = csr_matrix((Xtest, (Xtest_data['row'], Xtest_data['col'])),
                                  shape=(gatest.shape[0], napps))

        return Xtr_avgtime, Xte_avgtime


class TotalTimeOfAppsUsage(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'TotalTimeOfAppsUsage'
        self.description = 'Total time of application usage'

    def build(self):
        Xtrain_data = pd.read_csv(os.path.join(self._save_path, 'Xtrain_TimeOfAppsUsage.tsv'), delimiter='\t',
                                           names=['row', 'col', 'total', 'avg'])

        Xtest_data = pd.read_csv(os.path.join(self._save_path, 'Xtest_TimeOfAppsUsage.tsv'), delimiter='\t',
                                          names=['row', 'col', 'total', 'avg'])

        napps = 35000
        onehot_encoding = False

        if onehot_encoding:
            # one-hot encoding
            Xtrain = np.ones((Xtrain_data['total'].shape[0],), dtype=np.int)
            Xtest = np.ones((Xtest_data['total'].shape[0],), dtype=np.int)
        else:
            # weighted encoding
            Xtrain = Xtrain_data['total']
            Xtest = Xtest_data['total']

        Xtr_time = csr_matrix((Xtrain, (Xtrain_data['row'], Xtrain_data['col'])),
                                  shape=(gatrain.shape[0], napps))
        Xte_time = csr_matrix((Xtest, (Xtest_data['row'], Xtest_data['col'])),
                                  shape=(gatest.shape[0], napps))

        return Xtr_time, Xte_time


class AvgTimeOfAppCategoriesUsage(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'AvgTimeOfAppCategoriesUsage'
        self.description = 'Average time of application categories usage'

    def build(self):
        Xtrain_data = pd.read_csv(os.path.join(datadir, 'Xtrain_TimeOfAppCategoriesUsage.tsv'), delimiter='\t',
                                              names=['row', 'col', 'val'])

        Xtest_data = pd.read_csv(os.path.join(datadir, 'Xtest_TimeOfAppCategoriesUsage.tsv'), delimiter='\t',
                                             names=['row', 'col', 'val'])

        napps = 289

        onehot_encoding = False

        if onehot_encoding:
            # one-hot encoding
            Xtrain = np.ones((Xtrain_data['val'].shape[0],), dtype=np.int)
            Xtest = np.ones((Xtest_data['val'].shape[0],), dtype=np.int)
        else:
            # weighted encoding
            Xtrain = Xtrain_data['val']
            Xtest = Xtest_data['val']

        Xtr_avgtime = csr_matrix(
            (Xtrain, (Xtrain_data['row'], Xtrain_data['col'])),
            shape=(gatrain.shape[0], napps))
        Xte_avgtime = csr_matrix((Xtest, (Xtest_data['row'], Xtest_data['col'])),
                                     shape=(gatest.shape[0], napps))

        return Xtr_avgtime, Xte_avgtime


class KindsOfAppUsagePerDay(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'KindsOfAppUsagePerDay'
        self.description = 'Kinds of Application Usage per Day'

    def build(self):
        Xtrain_data = pd.read_csv(os.path.join(self._save_path, 'Xtrain_KindsOfAppUsagePerDay.tsv'), delimiter='\t',
                                              names=['device_id', 'kinds'])

        Xtest_data = pd.read_csv(os.path.join(self._save_path, 'Xtest_KindsOfAppUsagePerDay.tsv'), delimiter='\t',
                                             names=['device_id', 'kinds'])

        return csr_matrix(Xtrain_data['kinds']).T, csr_matrix(Xtest_data['kinds']).T


class KindsOfAppCatUsagePerDay(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'KindsOfAppCatUsagePerDay'
        self.description = 'Kinds of Application Categories Usage per Day'

    def build(self):
        Xtrain_data = pd.read_csv(os.path.join(self._save_path, 'Xtrain_KindsOfAppCatUsagePerDay.tsv'), delimiter='\t',
                                              names=['device_id', 'kinds'])

        Xtest_data = pd.read_csv(os.path.join(self._save_path, 'Xtest_KindsOfAppCatUsagePerDay.tsv'), delimiter='\t',
                                             names=['device_id', 'kinds'])

        return csr_matrix(Xtrain_data['kinds']).T, csr_matrix(Xtest_data['kinds']).T


class NumOfProcessUsagePerDay(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'NumOfProcessUsagePerDay'
        self.description = 'Number of process usage per day'

    def build(self):
        Xtrain_data = pd.read_csv(os.path.join(self._save_path, 'Xtrain_NumOfProcessUsagePerDay.tsv'), delimiter='\t',
                                              names=['device_id', 'num'])

        Xtest_data = pd.read_csv(os.path.join(self._save_path, 'Xtest_NumOfProcessUsagePerDay.tsv'), delimiter='\t',
                                             names=['device_id', 'num'])

        return csr_matrix(Xtrain_data['num']).T, csr_matrix(Xtest_data['num']).T


class TimeDistOfUserActivities(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'TimeDistOfUserActivities'
        self.description = 'Time distribution of user activities'

    def build(self):
        names = ['device_id']
        names.extend(range(0, 24))
        names.append('most_active_hour')
        Xtrain_data = pd.read_csv(os.path.join(self._save_path, 'Xtrain_TimeDistOfUserActivities.tsv'), delimiter=',',
                                              names=names)
        del Xtrain_data['device_id']
        del Xtrain_data['most_active_hour']

        Xtest_data = pd.read_csv(os.path.join(self._save_path, 'Xtest_TimeDistOfUserActivities.tsv'), delimiter=',',
                                 names=names)
        del Xtest_data['device_id']
        del Xtest_data['most_active_hour']

        return csr_matrix(Xtrain_data), csr_matrix(Xtest_data)


class MostActiveHour(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'MostActiveHour'
        self.description = 'Most Active Hour'

    def build(self):
        names = ['device_id']
        names.extend(range(0, 24))
        names.append('most_active_hour')
        
        Xtrain_data = pd.read_csv(os.path.join(self._save_path, 'Xtrain_TimeDistOfUserActivities.tsv'), delimiter=',',
                                              names=names)
        Xtr = Xtrain_data['most_active_hour']

        Xtest_data = pd.read_csv(os.path.join(self._save_path, 'Xtest_TimeDistOfUserActivities.tsv'), delimiter=',',
                                 names=names)
        Xte = Xtest_data['most_active_hour']

        return csr_matrix(Xtr).T, csr_matrix(Xte).T


class DayDistOfUserActivities(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'DayDistOfUserActivities'
        self.description = 'Day distribution of user activities'

    def build(self):
        names = ['device_id']
        names.extend(range(0, 7))
        names.append('most_active_day')
        Xtrain_data = pd.read_csv(os.path.join(self._save_path, 'Xtrain_DayDistOfUserActivities.tsv'), delimiter=',',
                                              names=names)
        del Xtrain_data['device_id']
        del Xtrain_data['most_active_day']

        Xtest_data = pd.read_csv(os.path.join(self._save_path, 'Xtest_DayDistOfUserActivities.tsv'), delimiter=',',
                                 names=names)
        del Xtest_data['device_id']
        del Xtest_data['most_active_day']

        return csr_matrix(Xtrain_data), csr_matrix(Xtest_data)


class MostActiveDay(BaseFeature):
    def __init__(self):
        super().__init__()
        self.name = 'MostActiveDay'
        self.description = 'Most Active Day'

    def build(self):
        names = ['device_id']
        names.extend(range(0, 7))
        names.append('most_active_day')

        Xtrain_data = pd.read_csv(os.path.join(self._save_path, 'Xtrain_DayDistOfUserActivities.tsv'), delimiter=',',
                                  names=names)
        Xtr = Xtrain_data['most_active_day']

        Xtest_data = pd.read_csv(os.path.join(self._save_path, 'Xtest_DayDistOfUserActivities.tsv'), delimiter=',',
                                 names=names)
        Xte = Xtest_data['most_active_day']

        return csr_matrix(Xtr).T, csr_matrix(Xte).T

    
features_code = {
    'Brand': 'F1',
    'Model': 'F2',
    'InstalledApps': 'F3',
    'InstalledAppsNumber': 'F4',
    'BagOfAppCategories_TFIDF': 'F5',
    'AvgTimeOfAppsUsage': 'F6',
    'TotalTimeOfAppsUsage': 'F7',
    'AvgTimeOfAppCategoriesUsage': 'F8',
    'KindsOfAppUsagePerDay': 'F9',
    'KindsOfAppCatUsagePerDay': 'F10',
    'NumOfProcessUsagePerDay': 'F11',
    'TimeDistOfUserActivities': 'F12',
    'MostActiveHour': 'F13',
    'DayDistOfUserActivities': 'F14',
    'MostActiveDay': 'F15'
}


def load_feature(feat_name):
    klass = globals()[feat_name]
    instance = klass()
    return instance.load()


def load_features(feats_name):
    Xtrain = []
    Xtest = []
    for name in feats_name:
        klass = globals()[name]
        instance = klass()
        Xtr, Xte = instance.load()
        Xtrain.append(Xtr)
        Xtest.append(Xte)

    Xtr = hstack(Xtrain)
    Xte = hstack(Xtest)

    print('All features: train shape {}, test shape {}'.format(Xtr.shape, Xte.shape))

    return Xtr, Xte


def get_feature_code(feat_name):
    return features_code[feat_name]
