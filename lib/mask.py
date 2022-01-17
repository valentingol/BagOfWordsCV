import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour


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


def init_contour(img, padding, nb_points):
    '''
    A function that initializes a contour for our dataset of images. 
    They're all framed approximately the same way so it would be very inefficient to initialize the algorithm to a different contour for each image.
    Since apples and tomatoes take almost all the space in the image, we will initialize a contour just a bit tighter than the borders of the images. 
    We'll call the distance to the borders "padding".

    Parameters
    ----------
    img : np.array
        Image to mask.
    padding: int
        The distance from the borders of the picture for the contour.
    nb_points: int
        The number of points on each of our segments of the picture. With nb_points, we have a contour of 4 * nb_points points.

    Returns
    -------
    contour: (np.array, np.array), dtype=np.uint8
        The initial contour to run the active contour algorithm on our images.

    Example
    -------
    If the img.shape == (w,h) and padding == n (provided that n < min(0.5*w, 0.5*h)) our contour has 4 segments:
    | y = n and x in [n, w-n]
    | x = w-n and y in [n, h-n]
    | y = h-n and x in [w-n, n]
    | x = n and y in [h-n, n]
    '''
    h, w = img[..., 0].shape
    step_x = (w - 2 * padding) // nb_points
    step_y = (h - 2 * padding) // nb_points

    segment1_x = np.arange(padding, w-padding, step_x)
    segment1_y = padding * np.ones(segment1_x.shape).astype(np.uint8)

    segment2_y = np.arange(padding, h-padding, step_y)
    segment2_x = (w-padding) * np.ones(segment2_y.shape).astype(np.uint8)

    segment3_x = np.flip(segment1_x)
    segment3_y = (h-padding) * np.ones(segment3_x.shape).astype(np.uint8)

    segment4_y = np.flip(segment2_y)
    segment4_x = padding * np.ones(segment4_y.shape).astype(np.uint8)

    contour_x = np.concatenate(
        (segment1_x, segment2_x, segment3_x, segment4_x), axis=0)
    contour_y = np.concatenate(
        (segment1_y, segment2_y, segment3_y, segment4_y), axis=0)

    return np.array([contour_x, contour_y])


def mask_active_contour(img, initial_contour):
    '''
    A function that computes the active contour of an image.
    [TO DO] apply a mask on everything that's outside the perimeter defined by the contour.
    '''
    img = rgb2gray(img)
    return active_contour(gaussian(img, 10, preserve_range=False),
                          initial_contour, alpha=.001, beta=.01, w_line=.1, w_edge=-.1, gamma=.001)


def test_active_contour(data):
    '''
    A function to compute and plot the active contour on an image of our dataset, to quickly assess the performances of the algorithm.
    '''
    _, ax = plt.subplots(figsize=(7, 7))
    img = data[0]
    initial_contour = init_contour(img, 25, 50)
    snake = mask_active_contour(img, initial_contour)

    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(initial_contour[0], initial_contour[1], '--r', lw=1)
    ax.plot(snake[0], snake[1], '--b', lw=4)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()


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
    import matplotlib.pyplot as plt

    infos = {'apple_a': (0, 100), 'apple_b': (1, 0), 'apple_c': (1, 0),
             'tomato': (1, 0)}
    (data, _), _ = get_dataset('dataset/fruits', infos)

    mask_thresholds = {'r': (0.2, '>'), 'g': (0.43, '<'), 'b': (0.39, '<')}
    mask_func = partial(mask_rgb, thresholds=mask_thresholds)

    # test_mask(data, mask_func, n=5)
    # test_mask(data, mask_apple_tomato, n=2)
    # test_mask(data, mask_otsu, n=2)
    test_active_contour(data)
