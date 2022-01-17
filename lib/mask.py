import cv2
import numpy as np
from skimage.filters import threshold_otsu


def mask_rgb(img, thresholds):
    """Mask the input image using RGB thresholds.

    Parameters
    ----------
    img : np.array
        Image to mask.
    thresholds: dict
        A dictionary of the form {'r': (threshold, '>' or '<')}, 'g': ...,
        'b': ...}. Where threshold is the threshold value (between 0 and 1)
        and '>' or '<' is whether to mask pixels with a value greater or lower
        than the threshold. All points has to check all the thresholds
        to be preserved.

    Returns
    -------
    mask_bool: np.array, dtype=bool
        Mask of the input image.
    """
    masks = []
    for i, color in enumerate(['r', 'g', 'b']):
        if thresholds[color][1] == '>':
            masks.append(img[..., i] > thresholds[color][0])
        elif thresholds[color][1] == '<':
            masks.append(img[..., i] < thresholds[color][0])
        else:
            raise ValueError(f"thresholds[{color}][1] must be '>' or '<', "
                             f"found {thresholds[color][1]}")
    mask_bool = np.logical_and(np.logical_and(masks[0], masks[1]), masks[2])
    return mask_bool


def mask_apple_tomato(img):
    mask_bool = img[..., 0] > (img[..., 1] + img[..., 2]) * 0.6
    return mask_bool


def to_grayscale(img):
    '''
    A grayscale conversion for images in numpy arrays.
    '''
    return img[..., :3] @ [0.299, 0.587, 0.114]


def mask_otsu(img):
    '''
    A function based on skimage.filters.threshold_otsu that applies a mask to the image given as parameter according to OTSU's method.
    '''
    gray_img = to_grayscale(img)
    thresh = threshold_otsu(gray_img)
    return gray_img > thresh


def test_mask(data, mask_func, n=20):
    """Display n images with their masks (computed using mask_func)."""
    for i, img in enumerate(data):
        mask = mask_func(img)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Two cases for masks, they either have two or three dimensions.
        masked_img = img * \
            mask[..., None] if len(mask.shape) < 3 else img * mask
        cv2.imshow(f'img {i}', img)
        cv2.imshow(f'masked img {i}', masked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if i == n - 1:
            break


if __name__ == '__main__':
    from functools import partial
    from lib.data import get_dataset

    infos = {'apple_a': (0, 100), 'apple_b': (1, 0), 'apple_c': (1, 0),
             'tomato': (1, 0)}
    (data, _), _ = get_dataset('dataset/fruits', infos)

    mask_thresholds = {'r': (0.2, '>'), 'g': (0.43, '<'), 'b': (0.39, '<')}
    mask_func = partial(mask_rgb, thresholds=mask_thresholds)

    # test_mask(data, mask_func, n=5)
    # test_mask(data, mask_apple_tomato, n=2)
    test_mask(data, mask_otsu, n=2)
