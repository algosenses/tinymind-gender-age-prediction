from abc import ABCMeta, abstractmethod
import os
from scipy.sparse import isspmatrix_csr, isspmatrix_csc
from scipy.sparse import save_npz, load_npz

class BaseFeature(metaclass=ABCMeta):
    def __init__(self):
        self.name = None
        self.description = 'Base class of features.'
        self.config = {}
        self._save_path = './features'
        if not os.path.exists(self._save_path):
            os.mkdir(self._save_path)

    @abstractmethod
    def build(self):
        raise NotImplementedError("Must override function 'build'")

    def load(self):
        Xtrain_file = os.path.join(self._save_path, ('Xtrain_' + self.name + '.npz'))
        Xtest_file = os.path.join(self._save_path, ('Xtest_' + self.name + '.npz'))

        if not os.path.isfile(Xtrain_file) or not os.path.isfile(Xtest_file):
            Xtrain, Xtest = self.build()

            if isspmatrix_csr(Xtrain) or isspmatrix_csc(Xtrain):
                save_npz(Xtrain_file, Xtrain)

            if isspmatrix_csr(Xtest) or isspmatrix_csc(Xtest):
                save_npz(Xtest_file, Xtest)
        else:
            Xtrain = load_npz(Xtrain_file)
            Xtest = load_npz(Xtest_file)

        print('Feature: [{}], Train shape: [{}], Test shape: [{}]'.format(self.description, Xtrain.shape, Xtest.shape))

        return Xtrain, Xtest
