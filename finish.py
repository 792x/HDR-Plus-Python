import cv2 as cv
import numpy as np
from datetime import datetime

from utils import time_diff


'''
Step 3 of HDR+ pipeline: finish

image : numpy ndarray
    The merged image to be finished

Returns: numpy ndarray (finished image)
'''
def finish_image(imgs, width, height, black_point, white_point, white_balance, compression, gain):
    print(f'\n{"="*30}\nFinishing image...\n{"="*30}')
    start = datetime.utcnow()

    # TODO

    print(f'Finishing finished in {time_diff(start)} ms.\n')
    return imgs