import cv2 as cv
import numpy as np
from datetime import datetime
import multiprocessing

from utils import time_diff


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
        pyramid = []
        for level in range(4):
            pyramid.append(cv.pyrDown(image) if level == 0 else cv.pyrDown(pyramid[level-1]))
        
        # TODO
    
    print(f'Alignment finished in {time_diff(start)} ms.\n')
    return images