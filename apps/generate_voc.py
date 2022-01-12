from time import time

from lib.bag_of_words import bag_of_words_kmeans
from lib.data import get_dataset
from lib.descriptors import get_keypoints_orb
from lib.mask import mask_apple_tomato

if __name__ == '__main__':
    n_clusters = 20
    descriptor = 'orb'
    save_path = f'vocabulary/fruits/tomato_{n_clusters}_{descriptor}.npy'

    infos = {'apple_a': (0, 0), 'apple_b': (1, 0), 'apple_c': (1, 0),
             'tomato': (1, 500)}

    (data_train, _), _ = get_dataset('dataset/fruits', infos)

    mask_func = mask_apple_tomato

    t0 = time()
    descriptors, _ = get_keypoints_orb(data_train, select=None,
                                        mask_func=mask_func)
    print('number of descriptors:', len(descriptors))

    # Get the vocabulary and save it
    bag_of_words_kmeans(descriptors, n_clusters=n_clusters, seed=0,
                        save_path=save_path)

    t1 = time()
    print('generation of bag of words done')
    print(f'time: {t1 - t0: .3f}s')
