import cv2 as cv
import numpy as np


'''
Overlays images by averaging the value of each pixel
Used to check alignment

images : list of numpy ndarray
    The aligned burst images to be averaged

Returns: numpy ndarray (average of the given images)
'''
def average_image(images):
    return np.mean(np.array(images), axis=0)


'''
Step 2 of HDR+ pipeline: merge

images : list of numpy ndarray
    Aligned burst images to be merged

Returns: numpy ndarray (merged image)
'''
def merge_images(images):
    print('Merging images...')

    # TODO

    return average_image(images)