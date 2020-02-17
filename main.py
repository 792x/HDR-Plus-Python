import cv2 as cv
import numpy as np
import rawpy
import imageio
import math
import os
import sys
import multiprocessing
import halide as hl
from datetime import datetime

os.environ['KIVY_NO_CONSOLELOG'] = '1'
import kivy
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.logger import Logger

from utils import time_diff

from align import align_images
from merge import merge_images
from finish import finish_image


'''
Loads a raw image

image_path : str
    String representing the path to the image

Returns: numpy ndarray with 3 values for each pixel
'''
def load_image(image_path):
    with rawpy.imread(image_path) as raw:
        image = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
        return image


'''
Converts a raw image to grayscale

image : numpy ndarray
    The image to be converted

Returns: numpy ndarray, where each pixel has one value (average of 3 given values)
'''
def to_grayscale(image):
    return np.mean(image, axis=2)


'''
Loads a burst of images

burst_path : str
    String representing the path to the folder containing the burst images

Returns: list of raw images, list of grayscale images, reference image
'''
def load_images(burst_path):
    print(f'\n{"="*30}\nLoading images...\n{"="*30}')
    start = datetime.utcnow()
    images = []
    
    # Create list of paths to the images
    paths = []
    for i in range(100):
        if i < 10:
            filename = f'payload_N00{i}.dng'
        else:
            filename = f'payload_N0{i}.dng'
        file_path = f'{burst_path}/{filename}'
        if os.path.isfile(file_path):
            paths.append(file_path)
        else:
            if i == 0:
                raise ValueError
            break
    
    # Load raw images
    print('Loading raw images...')
    p = multiprocessing.Pool(min(multiprocessing.cpu_count()-1, len(paths)))
    for image in p.imap(load_image, paths):
        images.append(hl.Buffer(image))

    assert len(images) >= 2, "Burst must consist of at least 2 images"

    # Get a reference image to compare results
    print('Getting reference image...')
    with rawpy.imread(paths[0]) as raw:
        ref_img = raw.postprocess()

    print('Building image buffer...')
    result = hl.Buffer(hl.Int(16), [images[0].width(), images[0].height(), len(images)])
    for index, image in enumerate(images):
        print(index)
        resultSlice = result.sliced(2, index)
        print(resultSlice)
        resultSlice.copy_from(image)

    print(f'Loading finished in {time_diff(start)} ms.\n')
    return result, ref_img


'''
Main method of the HDR+ pipeline: align, merge, finish

burst_path : str
    The path to the folder containing the burst images

Returns: str, str (paths to the reference and HDR images, respectively)
'''
def HDR(burst_path, black_point, white_point, white_balance, compression, gain):
    try:
        start = datetime.utcnow()

        # Load the images
        images, ref_img = load_images(burst_path)

        # dimensions of image should be 3
        assert images.dimensions() == 3, f"Incorrect buffer dimensions, expected 3 but got {images.dimensions()}"

        # Save the reference image
        imageio.imsave('Output/input.jpg', ref_img)

        # Align the images
        alignment = align_images(images)

        # Merge the images
        # merged = merge_images(imgs, alignment)

        # Finish the image
        # finished = finish_image(merged, width, height, black_point, white_point, white_balance, compression, gain)

        # TODO: replace with finished image rather than brighter
        # brighter = hl.Func("brighter")
        # x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")
        # brighter[x, y, c] = hl.cast(hl.UInt(8), hl.min(images[0][x, y, c] * 1.5, 255))
        # output_image = brighter.realize(images[0].width(), images[0].height(), images[0].channels())
        # imageio.imsave('Output/output.jpg', output_image)

        print(f'Processed in: {time_diff(start)} ms')

        return 'Output/input.jpg', 'Output/output.jpg'

    except Exception as e:
        print(e)
        # On error, return the empty gallery images
        return 'Images/gallery.jpg', 'Images/gallery.jpg'


class Imglayout(FloatLayout):
    def __init__(self, **args):
        super(Imglayout, self).__init__(**args)

        with self.canvas.before:
            Color(0, 0, 0, 0)
            self.rect = Rectangle(size=self.size,pos=self.pos)

        self.bind(size=self.updates,pos=self.updates)

    def updates(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(FloatLayout):
    loadfile = ObjectProperty(None)

    # Empty gallery images
    original = 'Images/gallery.jpg'
    image = 'Images/gallery.jpg'

    # Path to the burst images
    path = ''

    def build():
        c = Imglayout()
        root.add_widget(c)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        # Function to call the HDR+ pipeline
        def HDR_callback(instance):
            original_path, image_path = HDR(self.path, 0, 0, 0, 0, 0)
            self.original = original_path
            self.image = image_path
            self.ids.image0.source = self.original
            self.ids.image0.reload()
            self.ids.image1.source = self.image
            self.ids.image1.reload()

        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select burst image", content=content,
                            size_hint=(0.9, 0.9))

        # When a file is chosen, call the HDR+ pipeline
        self._popup.bind(on_dismiss=HDR_callback)

        self._popup.open()

    def load(self, path, filename):
        # Set the path to the burst images
        self.path = path
        
        self.dismiss_popup()


class HDR_Plus(App):
    pass


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)


if __name__ == '__main__':
    HDR_Plus().run()