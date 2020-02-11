import cv2 as cv
import numpy as np
from datetime import datetime
import multiprocessing

from utils import time_diff, Point


'''
Get a value representing the sharpness of an image

image : numpy ndarray
    The image for which to calculate the sharpness

Returns: float
'''
def sharpness(image):
    # Returns the average variance of the edges in an image
    # The higher the value, the sharper the image
    return cv.Laplacian(image, cv.CV_64F).var()


'''
Downsample an image by a factor of 2 in width and height

image : numpy.ndarray
    The image to be downsampled

Returns: numpy.ndarray
'''
def downsample(image):
    return cv.resize(image, None, fx=0.5, fy=0.5)


'''
Determines the best offset for tiles of the image at a given resolution, 
provided the offsets for the layer above

layer : numpy.ndarray
    The layer for which the offset needs to be calculated
prev_alignment : Point
    Alignment of the previous layer
prev_min : Point
    Min search region
prev_max : Point
    Max search region

Returns: Point
'''
def align_layer(layer, prev_alignment, prev_min, prev_max):

    # Inspiration from https://github.com/timothybrooks/hdr-plus/blob/master/src/align.cpp

    # TODO

    return prev_alignment


'''
Step 1 of HDR+ pipeline: align

images : list of numpy ndarray
    The raw burst images
grayscale : list of numpy ndarray
    Grayscale versions of the images

Returns: list of numpy ndarray (aligned images)
'''
def align_images(images, grayscale):
    print(f'\n{"="*30}\nAligning images...\n{"="*30}')
    start = datetime.utcnow()

    # Choose sharpest frame from the first 3 
    # frames of the burst as reference frame
    sharpness_list = []
    print('Choosing reference frame...')
    for i in range(3):
        sharpness_list.append(sharpness(images[i]))
        print(f'Frame {i}: sharpness = {sharpness_list[i]}')
    max_sharpness = 0
    sharpest = 0
    for i, x in enumerate(sharpness_list):
        if x > max_sharpness:
            max_sharpness = x
            sharpest = i
    reference_frame = images[sharpest]
    reference_grayscale = grayscale[sharpest]
    del images[sharpest]
    del grayscale[sharpest]
    print(f'Picked reference frame: {sharpest}')

    # Downsample grayscale images
    # Average 2 x 2 blocks
    print('Downsampling grayscale images...')
    downsampled_grayscale = []
    for image in grayscale:
        downsampled_grayscale.append(downsample(image))
    reference_downsampled_grayscale = downsample(reference_grayscale)

    # Hierarchical alignment
    for image in downsampled_grayscale:
        # 4-level gaussian pyramid
        # Each consecutive level has a lower resolution than the previous one
        pyramid = [image]
        for level in range(1,4):
            pyramid.append(cv.pyrDown(pyramid[level-1]))

        # Inspiration from https://github.com/timothybrooks/hdr-plus/blob/master/src/align.cpp

        downsample_rate = 2 # Default value in OpenCV

        min_search = Point(-4, -4)
        max_search = Point(3, 3)

        min_3 = Point(0, 0)
        min_2 = downsample_rate * min_3 + min_search
        min_1 = downsample_rate * min_2 + min_search

        max_3 = Point(0, 0)
        max_2 = downsample_rate * max_3 + max_search
        max_1 = downsample_rate * max_2 + max_search

        # initial alignment of previous layer is 0, 0
        alignment_3 = Point(0, 0)

        # Hierarchical alignment functions
        alignment_2 = align_layer(pyramid[1], alignment_3, min_3, max_3)
        alignment_1 = align_layer(pyramid[2], alignment_2, min_2, max_2)
        alignment_0 = align_layer(pyramid[3], alignment_1, min_1, max_1)
        
        # TODO
    
    print(f'Alignment finished in {time_diff(start)} ms.\n')
    return images