#%%
from functools import partial

import numpy as np

from lib.data import get_dataset
from lib.classifier import SVM
from lib.bag_of_words import get_features
from lib.descriptors import get_keypoints_sift
from lib.mask import mask_rgb


# Take dataset with same number of images on each class (apple and tomato)
infos = {'apple_a': (0, 300), 'apple_b': (1, 0), 'apple_c': (1, 0),
                'tomato': (1, 300)}
(data_train, y_train), (data_val, y_val) = get_dataset('dataset/fruits',
                                                    infos, val_ratio=0.2)

# Get vocabulary
vocabulary_apple = np.load('vocabulary/fruits/apple_a_200.npy')
vocabulary_tomato = np.load('vocabulary/fruits/tomato_200.npy')
vocabulary = np.concatenate((vocabulary_apple, vocabulary_tomato), axis=0)

# Get masking function
mask_thresholds = {'r': (0.4, '>'), 'g': (0.5, '<'), 'b': (0.5, '<')}
mask_func = partial(mask_rgb, thresholds=mask_thresholds)

# Get descriptors and make bag of words features
desc_train, _ = get_keypoints_sift(data_train, select=2000,
                                mask_func=mask_func)
desc_val, _ = get_keypoints_sift(data_val, select=2000,
                                mask_func=mask_func)
X_train = get_features(desc_train, vocabulary)
X_val = get_features(desc_val, vocabulary)

#%%
# Gridsearch for SVM
for C in [10]:
    for gamma in [2e-4]:
        svm = SVM(C=C, kernel='rbf', gamma=gamma, verbose=False)
        svm.fit(X_train, y_train)
        preds = svm.predict(X_val)
        accuracy = np.mean(preds == y_val)
        print(f'C={C}, gamma={gamma:.6f} - accuracy={100 * accuracy:.2f}%')

# %%
