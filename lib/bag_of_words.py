from functools import partial

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from lib.data import get_dataset
from lib.descriptors import get_keypoints_sift
from lib.mask import mask_rgb


def bag_of_words_kmeans(descriptors, n_clusters, seed=0, save_path=None):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(descriptors)
    vocabulary = kmeans.cluster_centers_
    if save_path is not None:
        np.save(save_path, vocabulary)
    return vocabulary


def bag_of_words_agglomerative_clustering(descriptors, n_clusters, save_path=None):
    agg_clustering = AgglomerativeClustering(
        n_clusters=n_clusters).fit(descriptors)
    labels = agg_clustering.labels_

    # We had cluster centers in kmeans, in this case we'll take random descriptors in the clusters as vocabulary words
    possible_labels = set(labels)
    first_occurences_of_labels = [
        np.where(labels == label)[0] for label in possible_labels]
    vocabulary = np.array([descriptors[index]
                          for index in first_occurences_of_labels])
    if save_path is not None:
        np.save(save_path, vocabulary)
    return vocabulary


def select_word(descriptors, vocabulary, dist, threshold=250_000_000):
    if dist == 'euclidian':
        ax = np.newaxis
        batch_size, n_desc, dim = descriptors.shape
        vocab_size = vocabulary.shape[0]
        if batch_size * n_desc * vocab_size * dim > threshold:
            vocabulary = vocabulary[ax, ...]
            dist = np.empty((batch_size, n_desc, vocab_size))
            for i, desc_img in enumerate(descriptors):
                desc_img = desc_img[:, ax, :]
                dist[i] = np.sum(
                    np.square(desc_img - vocabulary), axis=-1) ** 0.5
        else:
            descriptors = descriptors[..., ax, :]
            vocabulary = vocabulary[ax, ax, ...]
            dist = np.sum(np.square(descriptors - vocabulary), axis=-1) ** 0.5
    elif dist == 'cosine':
        # Cosine distance
        dist = np.dot(descriptors, vocabulary.T)
        dist /= np.linalg.norm(descriptors, axis=-1)[..., None]
        dist /= np.linalg.norm(vocabulary, axis=-1)[None, :]
    else:
        raise ValueError(f"Unknown distance: '{dist}' (should be 'euclidian' "
                         "or 'cosine'")
    return np.argmax(dist, axis=-1)


def get_features(descriptors, vocabulary, dist='euclidian'):
    batch_size = descriptors.shape[0]
    words = select_word(descriptors, vocabulary, dist=dist)
    features_list = []

    for i in range(batch_size):
        features = np.histogram(
            words[i], bins=len(vocabulary), density=True)[0]
        features_list.append(features * len(vocabulary))
    return np.array(features_list)


if __name__ == '__main__':
    n_clusters = 20

    infos = {'apple_a': (0, 20), 'apple_b': (1, 0), 'apple_c': (1, 0),
             'tomato': (1, 0)}
    (data_train, _), (data_val, _) = get_dataset('dataset/fruits', infos)

    mask_thresholds = {'r': (0.4, '>'), 'g': (0.5, '<'), 'b': (0.5, '<')}
    mask_func = partial(mask_rgb, thresholds=mask_thresholds)

    desc_train, _ = get_keypoints_sift(data_train, select=None,
                                       mask_func=mask_func)
    print('number of descriptors:', len(desc_train))

    vocabulary = bag_of_words_kmeans(desc_train, n_clusters=n_clusters, seed=0,
                                     save_path=None)

    desc_val, _ = get_keypoints_sift(data_val, select=200,
                                     mask_func=mask_func)
    # desc_val shape = (n_val, 200, desc_dim)
    n_val = len(desc_val)
    n_voc = len(vocabulary)
    fig, axes = plt.subplots(n_val//2, 2, figsize=(12, 6 * n_val))
    for i, descriptors in enumerate(desc_val):
        d = descriptors.shape[-1]
        descriptors = descriptors.reshape(1, -1, d)
        plt.sca(axes[i//2, i % 2])
        plt.title('image {}'.format(i))
        plt.xlabel('word')
        plt.ylabel('proportion')
        features = get_features(descriptors, vocabulary)
        features = features.reshape(-1, )
        plt.bar(np.arange(n_voc), features, width=1)
    plt.show()
