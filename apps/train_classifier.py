#%%
import numpy as np

from lib.data import get_dataset
from lib.classifier import SVM
from lib.bag_of_words import get_features
from lib.descriptors import get_keypoints_orb
from lib.mask import mask_apple_tomato

# Take dataset with same number of images on each class (apple and tomato)
infos = {'apple_a': (0, 500), 'apple_b': (1, 0), 'apple_c': (1, 0),
                'tomato': (1, 500)}
(data, y), (data_test, y_test) = get_dataset('dataset/fruits',
                                             infos, val_ratio=0.15)
print('Train + validation set:', len(data), 'samples.')
print('Test set:', len(data_test), 'samples.')


# Get vocabulary
vocabulary_apple = np.load('vocabulary/fruits/apple_a_20_orb.npy')
vocabulary_tomato = np.load('vocabulary/fruits/tomato_20_orb.npy')
vocabulary = np.concatenate((vocabulary_apple, vocabulary_tomato), axis=0)

# Get masking function
mask_func = mask_apple_tomato

#%%

# Get descriptors and make bag of words features
desc, _ = get_keypoints_orb(data, select=500,
                                mask_func=mask_func)
desc_test, _ = get_keypoints_orb(data_test, select=500,
                                mask_func=mask_func)
X = get_features(desc, vocabulary)
X_test = get_features(desc_test, vocabulary)
print('Features extracted.')

#%%
# Cross-validation and gridsearch for SVM
K = 5
for C in [50, 60, 70]:
    for gamma in [40e-4, 50e-4, 60e-4, 70e-4]:
        mean_acc = 0.0
        for k in range(K):
            X_val = X[k * len(X) // K:(k + 1) * len(X) // K]
            X_train = np.concatenate((X[:k * len(X) // K],
                                        X[(k + 1) * len(X) // K:]), axis=0)
            y_val = y[k * len(y) // K:(k + 1) * len(y) // K]
            y_train = np.concatenate((y[:k * len(y) // K],
                                        y[(k + 1) * len(y) // K:]), axis=0)
            svm = SVM(C=C, kernel='rbf', gamma=gamma, verbose=False)
            svm.fit(X_train, y_train)
            preds = svm.predict(X_val)
            mean_acc += np.mean(preds == y_val) / K
        print(f'C={C}, gamma={gamma:.6f} - val acc={100 * mean_acc:.2f}%')

# %%
# Test SVM on test set
C = 50
gamma = 0.007
svm = SVM(C=C, kernel='rbf', gamma=gamma, verbose=False)
svm.fit(X_train, y_train)
preds = svm.predict(X_test)
accuracy = np.mean(preds == y_test)
print(f'test accuracy: {100 * accuracy:.2f}%')
# %%
