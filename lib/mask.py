import cv2
import numpy as np

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


def test_mask(data, mask_func, n=20):
    """Display n images with their masks (computed using mask_func)."""
    for i, img in enumerate(data):
        mask = mask_func(img)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        masked_img = img * mask[..., None]

        cv2.imshow(f'img {i}', img)
        cv2.imshow(f'masked img {i}', masked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if i == n - 1:
            break


def mask_apple_tomato(imgs):
    masks_bool = imgs[..., 0] > (imgs[..., 1] + imgs[..., 2]) * 0.65
    masks = np.where(masks_bool, 1.0, 0.0)
    if masks_bool.ndim == 3:
        masks = cv2.dilate(masks, kernel=np.ones((5, 5), np.uint8), iterations=4)
        return np.squeeze(np.where(masks == 1.0, True, False))
    else:
        res = []
        for mask in masks:
            mask = cv2.dilate(mask, kernel=np.ones((5, 5), np.uint8), iterations=4)
            res.append(np.squeeze(np.where(mask == 1.0, True, False)))
    return np.array(res)

if __name__ == '__main__':
    from functools import partial
    from lib.data import get_dataset

    infos = {'apple_a': (0, 100), 'apple_b': (1, 0), 'apple_c': (1, 0),
                'tomato': (1, 0)}
    (data, _), _ = get_dataset('dataset/fruits', infos)

    mask_thresholds = {'r': (0.2, '>'), 'g': (0.43, '<'), 'b': (0.39, '<')}
    mask_func = partial(mask_rgb, thresholds=mask_thresholds)

    test_mask(data, mask_apple_tomato, n=5)
