import numpy as np
import sklearn


class SVM(object):
    def __init__(self, name='svm', **svm_params):
        super().__init__()
        self.name = name
        self.model = sklearn.svm.SVC(**svm_params)

    def fit(self, imgs, labels):
        self.model.fit(imgs, labels)

    def predict(self, imgs):
        return self.model.predict(imgs)

    def __call__(self, imgs):
        return self.predict(imgs)
