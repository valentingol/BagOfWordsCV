import os

import cv2
import numpy as np

def get_dataset(path, infos, val_ratio=0.2, seed=0, resolution=None):
    """Load the dataset from the given path.

    Parameters
    ----------
    path : str
        Path to the dataset.
    infos : dict
        A dictionary of the form {'subfolder_name': (label, num_sample)}
        Where subfolder_name is the name of the subfolder in the path
        containing pictures, label is the label of these pictures and
        num_sample is the number of pictures to take from the subfolder.
    val_ratio : float, optional
        Validation set ration, by default 0.2.
    seed : int, optional
        Seed of the shuffle operation, by default 0.
    resolution: tuple or None, optional
        Resolution of the pictures, by default None (no change).

    Returns
    -------
    (data_train, label_train), (data_val, label_val): np.array
        Training and validation sets.
    """
    # Set the seed
    np.random.seed(seed)

    datasets = os.listdir(path)
    images = {}
    for dataset in datasets:
        dataset_path = os.path.join(path, dataset)
        if os.path.isdir(dataset_path):
            label, n_sample = infos[dataset]
            if label not in images:
                images[label] =  []
            c = 0
            for fname in os.listdir(dataset_path):
                if fname.endswith('.jpg') or fname.endswith('.png'):
                    c += 1
                    img = cv2.imread(os.path.join(dataset_path, fname))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if resolution is not None:
                        img = cv2.resize(img, (resolution[1], resolution[0]))
                    images[label].append(img)
                    if c >= n_sample:
                        break
    data_train, data_val, label_train, label_val = [], [], [], []
    for label in images:
        data = np.array(images[label]) / 255.0
        np.random.shuffle(np.array(images[label]))
        n = data.shape[0]
        n_val = int(val_ratio * n)
        data_train.append(data[n_val:])
        data_val.append(data[:n_val])
        label_train.append(np.full(n - n_val, label))
        label_val.append(np.full(n_val, label))

    data_train, label_train = np.concatenate(data_train), np.concatenate(label_train)
    if data_train.ndim == 5:
        data_train = data_train.reshape((-1, *data_train.shape[2:]))
        label_train = label_train.reshape((-1, *label_train.shape[2:]))
    rd_perm = np.random.permutation(len(data_train))
    data_train, label_train = data_train[rd_perm], label_train[rd_perm]

    data_val, label_val = np.concatenate(data_val), np.concatenate(label_val)
    if data_val.ndim == 5:
        data_val = data_val.reshape((-1, *data_val.shape[2:]))
        label_val = label_val.reshape((-1, *label_val.shape[2:]))
    rd_perm = np.random.permutation(len(data_val))
    data_val, label_val = data_val[rd_perm], label_val[rd_perm]

    # Free the seed
    np.random.seed(None)

    return (data_train, label_train), (data_val, label_val)

if __name__ == '__main__':
    infos = {'apple_a': (0, 30), 'apple_b': (1, 10), 'apple_c': (1, 10),
             'tomato': (1, 10)}
    data = get_dataset('dataset/fruits', infos)
    (data_train, label_train), (data_val, label_val) = data
    print(data_train.shape, label_train.shape)
    print(data_val.shape, label_val.shape)
    print(label_val)