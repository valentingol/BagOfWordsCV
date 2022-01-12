import cv2
import numpy as np

from lib.data import get_dataset
from lib.mask import mask_apple_tomato


def get_keypoints_sift(data, select=None, mask_func=None,
                       show_keypoints=False, seed=0):
    """Get keypoints using SIFT.

    Parameters
    ----------
    data : np.array
        Images.
    select : int or None, optional
        Number of point to select from detected keypoints
        (selection at random). If None, all keypoints are kept,
        by default None.
    mask_func : callable or None, optional
        Mask function to select only keypoints on wanted area.
        No mask applied if None, by default None
    show_keypoints : bool, optional
        Whether to show the number of keypoints in each image,
        by default False.
    seed : int, optional
        Seed for random operation (not used if select is None),
        by default 0.

    Returns
    -------
    descriptors: np.array
        Descriptors of the keypoints. The shape is (all_keypoints, 128)
        if select is None and (n_images, select, 128) otherwise.
    keypoints: KeyPoint list
        List of keypoints of all the images. See OpenCV documentation
        for more information on KeyPoint.
    """
    # Set the seed
    np.random.seed(seed)

    descriptors = []
    keypoints = []
    for ind, img in enumerate(data):
        if mask_func is not None:
            img = img * mask_func(img)[..., None]

        # Get SIFT keypoints
        gray= cv2.cvtColor((img * 255).astype(np.uint8),
                           cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)

        indicies = np.arange(len(des))

        if select is not None:
            # Select a subset of keypoints (at random)
            k = select // len(indicies)
            indicies_list = []
            for _ in range(k):
                indicies_list.append(indicies)
            indicies_list.append(np.random.choice(indicies, select % len(indicies),
                                              replace=False))
            indicies = np.hstack(indicies_list)

        keypoints.append([kp[i] for i in indicies])
        if select is not None:
            descriptors.append([des[indicies]])
        else:
            descriptors.append(des[indicies])

        if show_keypoints:
            print(f'image {ind}: {len(indicies)} keypoints')
    descriptors = np.concatenate(descriptors, axis=0)
    # Free the seed
    np.random.seed(None)

    return descriptors, keypoints


def get_keypoints_orb(data, select=None, mask_func=None,
                       show_keypoints=False, seed=0):
    """Get keypoints using ORB.

    Parameters
    ----------
    data : np.array
        Images.
    select : int or None, optional
        Number of point to select from detected keypoints
        (selection at random). If None, all keypoints are kept,
        by default None.
    mask_func : callable or None, optional
        Mask function to select only keypoints on wanted area.
        No mask applied if None, by default None
    show_keypoints : bool, optional
        Whether to show the number of keypoints in each image,
        by default False.
    seed : int, optional
        Seed for random operation (not used if select is None),
        by default 0.

    Returns
    -------
    descriptors: np.array
        Descriptors of the keypoints. The shape is (all_keypoints, 32)
        if select is None and (n_images, select, 32) otherwise.
    keypoints: KeyPoint list
        List of keypoints of all the images. See OpenCV documentation
        for more information on KeyPoint.
    """
    # Set the seed
    np.random.seed(seed)

    descriptors = []
    keypoints = []
    for ind, img in enumerate(data):
        if mask_func is not None:
            img = img * mask_func(img)[..., None]

        # Get ORB keypoints
        gray= cv2.cvtColor((img * 255).astype(np.uint8),
                           cv2.COLOR_RGB2GRAY)
        surf = cv2.ORB_create()
        kp, des = surf.detectAndCompute(gray, None)

        indicies = np.arange(len(des))

        if select is not None:
            # Select a subset of keypoints (at random)
            k = select // len(indicies)
            indicies_list = []
            for _ in range(k):
                indicies_list.append(indicies)
            indicies_list.append(np.random.choice(indicies, select % len(indicies),
                                              replace=False))
            indicies = np.hstack(indicies_list)

        keypoints.append([kp[i] for i in indicies])
        if select is not None:
            descriptors.append([des[indicies]])
        else:
            descriptors.append(des[indicies])

        if show_keypoints:
            print(f'image {ind}: {len(indicies)} keypoints')
    descriptors = np.concatenate(descriptors, axis=0)
    # Free the seed
    np.random.seed(None)

    return descriptors, keypoints

if __name__ == '__main__':
    from time import time
    import matplotlib.pyplot as plt

    infos = {'apple_a': (0, 300), 'apple_b': (1, 100), 'apple_c': (1, 100),
             'tomato': (1, 100)}
    (data_train, _), _ = get_dataset('dataset/fruits', infos)

    mask_func = mask_apple_tomato

    t0 = time()
    desc, keyps = get_keypoints_sift(data_train, select=None, mask_func=None)
    t1 = time()
    print('descriptor shape (select=None)', desc.shape)

    desc, keyps = get_keypoints_orb(data_train, select=100,
                                    mask_func=mask_func)
    print('descriptor shape (select=100)', desc.shape)

    print(f'time to get keypoints: {t1 - t0: .2f}s')
    img = (data_train[0] * 255).astype(np.uint8)
    img2 = cv2.drawKeypoints(img, keyps[0], None, (255,0,0), 4)
    plt.imshow(img2)
    plt.show()
