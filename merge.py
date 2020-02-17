import cv2 as cv
import numpy as np
from datetime import datetime

from utils import time_diff


'''
Overlays images by averaging the value of each pixel
Used to check alignment

images : list of numpy ndarray
    The aligned burst images to be averaged

Returns: numpy ndarray (average of the given images)
'''
def average_image(imgs):
    return np.mean(np.array(imgs), axis=0)


'''
Step 2 of HDR+ pipeline: merge

images : list of numpy ndarray
    Aligned burst images to be merged

Returns: numpy ndarray (merged image)
'''
def merge_images(imgs, alignment):
    print(f'\n{"="*30}\nMerging images...\n{"="*30}')
    start = datetime.utcnow()

    # TODO

    print(f'Merging finished in {time_diff(start)} ms.\n')
    return imgs