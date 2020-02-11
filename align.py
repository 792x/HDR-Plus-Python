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
    p = multiprocessing.Pool(min(multiprocessing.cpu_count()-1, 3))
    for value in p.imap(sharpness, images[:3]):
        sharpness_list.append(value)
    for i in range(3):
        print(f'Frame {i}: sharpness = {sharpness_list[i]}')
    max_sharpness = 0
    sharpest = 0
    for i, x in enumerate(sharpness_list):
        if x > max_sharpness:
            max_sharpness = x
            sharpest = i
    reference_frame = images[sharpest]
    print(f'Picked reference frame: {sharpest}')

    # TODO
    
    print(f'Alignment finished in {time_diff(start)} ms.\n')
    return images