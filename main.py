import cv2 as cv
import numpy as np
import rawpy
import imageio
from matplotlib import pyplot as plt
import math
import os
import sys

dataset_path = 'C:/20171106_subset/bursts'
burst = f'{dataset_path}/4KK2_20150823_152106_985'

files = []

for i in range(100):
    if i < 10:
        filename = f'payload_N00{i}.dng'
    else:
        filename = f'payload_N0{i}.dng'
    path = f'{burst}/{filename}'
    try:
        with rawpy.imread(path) as raw:
            print(f'Reading file: \"{path}\"')
            image = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
            files.append(image)
    except:
        if i == 0:
            print(f'Burst format at \"{burst}\" not recognized.')
            sys.exit(0)
        else:
            break

def show_image(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.waitKey(0)

show_image(files[0])

# TODO