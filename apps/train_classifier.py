import numpy as np

from lib.data import get_dataset
from lib.classifier import SVM
from lib.bag_of_words import get_features
from lib.descriptors import get_keypoints_orb
from lib.mask import mask_apple_tomato

def get_data_and_voc(infos, val_ratio, voc_path1, voc_path2):
    (data, y), (data_test, y_test) = get_dataset('dataset/fruits',
                                                infos, val_ratio=val_ratio)
    vocabulary1 = np.load(voc_path1)
    vocabulary2 = np.load(voc_path2)
    vocabulary = np.concatenate((vocabulary1, vocabulary2), axis=0)
    return (data, y), (data_test, y_test), vocabulary


def random_cross_validation(n_folds, X, y):
    mean_acc = 0.0
    for _ in range(n_folds):
        val_idx = np.random.choice(len(X), int(len(X) * data_prop),
                                replace=False)
        train_idx = np.array([i for i in range(len(X)) if i not
                                in val_idx])
        X_val, y_val = X[val_idx], y[val_idx]
        X_train, y_train = X[train_idx], y[train_idx]
        svm = SVM(C=C, kernel='rbf', gamma=gamma, verbose=False)
        svm.fit(X_train, y_train)
        preds = svm.predict(X_val)
        mean_acc += np.mean(preds == y_val) / n_folds
    return mean_acc


def test_classifier(svm, X_train, y_train, X_test, y_test):
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    return np.mean(preds == y_test)


if __name__ == '__main__':
    ## Configs
    # Dataset / vocabulary / masking function
    infos = {'apple_a': (0, 500), 'apple_b': (1, 500), 'apple_c': (1, 0),
                    'tomato': (1, 0)}
    val_ratio = 0.15
    voc_path1 = 'vocabulary/fruits/apple_a_50_orb.npy'
    voc_path2 = 'vocabulary/fruits/apple_b_50_orb.npy'
    mask_func = mask_apple_tomato
    # Descriptors
    n_desc = 500
    dist = 'euclidian'
    # Cross validation
    data_prop = 0.2
    n_folds = 20
    C_range = [1, 10, 100, 1000]
    gamma_range = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
    # Test
    test = True
    test_svm = SVM(C=100, kernel='rbf', gamma=0.1, verbose=False)


    (data, y), (data_test, y_test), vocabulary = get_data_and_voc(
        infos, val_ratio, voc_path1, voc_path2
        )
    print('Train & validation set:', len(data), 'samples.')
    print('Test set:', len(data_test), 'samples.')

    # Get descriptors and make bag of words features
    desc, _ = get_keypoints_orb(data, select=n_desc,
                                    mask_func=mask_func)
    desc_test, _ = get_keypoints_orb(data_test, select=n_desc,
                                    mask_func=mask_func)
    X = get_features(desc, vocabulary)
    X_test = get_features(desc_test, vocabulary, dist=dist)
    print('Features extracted.')

    # Cross-validation and gridsearch for SVM
    for C in C_range:
        for gamma in gamma_range:
            mean_acc = random_cross_validation(n_folds, X, y)
            print(f'C={C}, gamma={gamma:.6f} - val acc={100 * mean_acc:.2f}%')

    if test:
        # Test SVM on test set
        accuracy = test_classifier(test_svm, X, y, X_test, y_test)
        print(f'test accuracy: {100 * accuracy:.2f}%')
