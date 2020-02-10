import cv2 as cv
import numpy as np
import rawpy
import imageio
import math
import os
import sys
import multiprocessing
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from datetime import datetime, timedelta

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
    return np.mean(image, axis=2, dtype=np.uint8)

'''
Loads a burst of images

burst_path : str
    String representing the path to the folder containing the burst images

Returns: list of raw images, list of grayscale images, reference image
'''
def load_images(burst_path):
    print('Loading images...')
    images = []
    grayscale = []
    
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
    p = multiprocessing.Pool(min(multiprocessing.cpu_count()-1, len(paths)))
    for image in p.imap_unordered(load_image, paths):
        images.append(image)
    
    # Convert images to grayscale
    p = multiprocessing.Pool(min(multiprocessing.cpu_count()-1, len(images)))
    for image in p.imap(to_grayscale, images):
        grayscale.append(image)

    # Get reference image
    with rawpy.imread(paths[0]) as raw:
        ref_img = raw.postprocess()

    return images, grayscale, ref_img

'''
Step 1 of HDR+ pipeline: align

images : list of numpy ndarray
    The raw burst images
grayscale : list of numpy ndarray
    Grayscale versions of the images

Returns: list of numpy ndarray (aligned images)
'''
def align_images(images, grayscale):
    print('Aligning images...')

    # TODO
 
    return images

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

'''
Step 3 of HDR+ pipeline: finish

image : numpy ndarray
    The merged image to be finished

Returns: numpy ndarray (finished image)
'''
def finish_image(image):
    print('Finishing image...')

    # TODO

    return image

'''
Main method of the HDR+ pipeline: align, merge, finish

burst_path : str
    The path to the folder containing the burst images

Returns: str, str (paths to the reference and HDR images, respectively)
'''
def HDR(burst_path):
    try:
        start = datetime.utcnow()

        # Load the images
        images, grayscale, ref_img = load_images(burst_path)

        # Save the reference image
        imageio.imsave('Output/input.jpg', ref_img)

        # Align the images
        images = align_images(images, grayscale)

        # Merge the images
        image = merge_images(images)

        # Finish the image
        image = finish_image(image)

        # Save the HDR image
        imageio.imsave('Output/output.jpg', image)

        time_dif = datetime.utcnow() - start
        print(f'Processed in: {time_dif.total_seconds()*1000} ms')

        return 'Output/input.jpg', 'Output/output.jpg'

    except Exception as e:
        print(e)
        # On error, return the empty gallery images
        return 'Images/gallery.jpg', 'Images/gallery.jpg'

class Imglayout(FloatLayout):
    def __init__(self,**args):
        super(Imglayout,self).__init__(**args)

        with self.canvas.before:
            Color(0,0,0,0)
            self.rect=Rectangle(size=self.size,pos=self.pos)

        self.bind(size=self.updates,pos=self.updates)

    def updates(self,instance,value):
        self.rect.size=instance.size
        self.rect.pos=instance.pos

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
            original_path, image_path = HDR(self.path)
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