# Bag-of-Words-Computer-Vision - a keypoints-based image classification pipeline with classic Computer Vision techniques

Computer Vision project for image classification pipeline using keypoints detection.

![intro](ressources/bag_of_words_intro.jpg)

## Description

The project follow the [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision) approach to image clasification.

The pipeline can be described as follows:

- 1. Apply a segmentation algorithm on all images to extract regions of interest

- 2. Apply a keypoints detection algorithm such as SIFT or SURF on images and select the keypoints in regions of interest

- 3. Apply a clustering algorithm on each keypoint descriptors of each class to extract a vocabulary of descriptors

- 4. Apply a histogram-of-words algorithm on each image to get features vectors

- 5. Apply a classifier like SVM or Gradient Boosting to classify each images

## Example on fruits dataset

The fruits dataset is available on the repository (loaded with Git LFS). The dataset provided is only a small part of the [original dataset](https://www.kaggle.com/chrisfilo/fruit-recognition) containing only images of size (322, 488), 3 kinds of red apples and tomatoes.

A simple segmentation using RGB thresholding is first applied to the images to get the mask of apple and tomatoes:

![simple segmentation 1](ressources/rgb_segmentation_1.png)
![simple segmentation 2](ressources/rgb_segmentation_2.png)

Then, the SIFT keypoints detection is applied on the gray-scale images and only the keypoints in the mask or kept:

![sift keypoint 1](ressources/sift_keypoints_1.jpg)
![sift keypoint 2](ressources/sift_keypoints_2.jpg)
![sift keypoint 3](ressources/sift_keypoints_3.jpg)

Then, the descriptors are extracted from the keypoints and clustered to get the vocabulary for each class.

After that, we can compute a vector of features for each image using SIFT descriptors and the vocabulary. For example with 4 images of apples we get the following histograms in a simple case were the vocabulary is composed of 20 words:

![features histograms](ressources/features_histograms.png)

Finally, a SVM is apply to classify the images correctly using the features vectors.

![SVM](ressources/img_svm.png)

## How to use the project

### Understand the functions in `lib` and implement your own (if needed)

- `lib/data.py`: functions to load and label your dataset

- `lib/mask.py`: functions to apply segmentation (before applying a keypoint detector)

- `lib/descriptors.py`: keypoints detector and extractor of their descriptions

- `lib/bag_of_words.py`: functions to generate a vocabulary

- `lib/classifier.py`: class of final classifiers (SVM, ...)

### Generate your own vocabularies and train your classifier

To generate your vocabularies using the functions in `lib`:

```script
python3 apps/generate_voc.py
```

To train your classifier (with grid search inside):

```script
python3 apps/train_classifier.py
```

## To-do list

Pipeline:

- [x] Implement a complete pipeline

- [x] Improve the cross-validation to less overfit the validation set

- [ ] Optimize the vocabulary size

Implement segmentation algorithm:

- [x] RGB thresholding

- [x] Otsu's method

- [x] Active contour segmentation

- [ ] Watershed segmentation

Implement keypoints detectors:

- [x] SIFT

- [ ] SURF

- [x] ORB

Implement vocabulary clustering algorithm:

- [x] K-mean

- [ ] PCA (to applied before clustering)

- [ ] Agglomerative clustering

Implement method to get features vectors:

- [x] Cosine distance 1-NN

- [x] Euclidian distance 1-NN

Implement classifier:

- [x] SVM
