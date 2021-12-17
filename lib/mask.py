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
    for i in range(n):
        img = data[i]
        mask = mask_func(img)
        cv2.imshow(f'img {i}', (img * 255).astype(np.uint8))
        cv2.imshow(f'mask {i}', (mask * 255).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
