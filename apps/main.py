from time import time

import cv2
import numpy as np

from lib.data import get_dataset

def mask_red(img):
    test_r = img[..., 0] > 0.4
    test_g = img[..., 1] < 0.5
    test_b = img[..., 1] < 0.5
    mask_bool = np.logical_and(np.logical_and(test_r, test_g), test_b)
    return mask_bool


def get_keypoints(data, select=None, show_img=False, show_keypoints=False,
                  seed=0):
    # Set the seed
    np.random.seed(seed)

    descriptors = []
    keypoints = []
    for ind, img in enumerate(data):
        mask = mask_red(img)
        if show_img and ind==393:
            cv2.imshow(f'img {ind}', (img * 255).astype(np.uint8))
            cv2.imshow(f'mask {ind}', (mask * 255).astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        gray= cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
        pts = np.array([kp[i].pt for i in range(len(kp))])
        pts = pts.astype(np.uint8)
        indicies = np.where(mask[pts[:, 1], pts[:, 0]])[0]
        if select is not None:
            indicies = np.random.choice(indicies, select, replace=True)
        if show_keypoints:
            print(f'image {ind}: {len(indicies)} keypoints')
        if len(indicies) < 20:
            print(f'image {ind}: {len(indicies)} keypoints')
        keypoints.append([kp[i] for i in indicies])
        if select is not None:
            descriptors.append([des[indicies]])
        else:
            descriptors.append(des[indicies])
    descriptors = np.concatenate(descriptors, axis=0)

    # Free the seed
    np.random.seed(None)

    return descriptors, keypoints

if __name__ == '__main__':
    infos = {'apple_a': (0, 300), 'apple_b': (1, 100), 'apple_c': (1, 100),
             'tomato': (1, 100)}
    (data_train, _), _ = get_dataset('dataset/fruits', infos)
    t0 = time()
    desc, keyps = get_keypoints(data_train, select=None, show_img=True)
    t1 = time()
    print('descriptor shape (select=None)', desc.shape)

    desc, keyps = get_keypoints(data_train, select=20)
    print('descriptor shape (select=20)', desc.shape)
    print('num keypoints', len(keyps))

    print(f'time to get keypoints: {t1 - t0: .2f}s')
