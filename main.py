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

def load_image(image_path):
    with rawpy.imread(image_path) as raw:
        image = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
        return image

def load_images(burst_path):
    print('Loading images...')
    images = []
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
    p = multiprocessing.Pool(min(multiprocessing.cpu_count()-1, len(paths)))
    for image in p.imap_unordered(load_image, paths):
        images.append(image)
    with rawpy.imread(paths[0]) as raw:
        ref_img = raw.postprocess()
    return images, ref_img

def align_images(images, grayscale):
    print('Aligning images...')

    # This sucks ↓ it is worse than no alignment at all
    # alignMTB = cv.createAlignMTB()
    # alignMTB.process(images[:2], images)
 
    return images

def to_grayscale(image):
    return np.mean(image, axis=2, dtype=np.uint8)

def average_image(images):
    avg_img = np.mean(np.array(images), axis=0)
    return avg_img

def merge_images(images):
    print('Merging images...')
    # TODO
    return average_image(images)

def finish_image(image):
    print('Finishing image...')
    # TODO
    return image

def HDR(burst_path):
    start = datetime.utcnow()
    try:
        images, ref_img = load_images(burst_path)
    except Exception as e:
        print(e)
        print(f'Burst format at \"{burst_path}\" not recognized.')
        return 'gallery.jpg', 'gallery.jpg'
    imageio.imsave('Output/input.jpg', ref_img)

    grayscale = []
    p = multiprocessing.Pool(min(multiprocessing.cpu_count()-1, len(images)))
    for image in p.imap_unordered(to_grayscale, images):
        grayscale.append(image)

    aligned = align_images(images, grayscale)

    merged = merge_images(aligned)

    image = finish_image(merged)

    imageio.imsave('Output/output.jpg', image)
    time_dif = datetime.utcnow() - start
    print(f'Processed in: {time_dif.total_seconds()*1000} ms')
    return 'Output/input.jpg', 'Output/output.jpg'

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
    original = 'gallery.jpg'
    image = 'gallery.jpg'
    path = ''
    def build():
        c = Imglayout()
        root.add_widget(c)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
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
        self._popup.bind(on_dismiss=HDR_callback)
        self._popup.open()

    def load(self, path, filename):
        self.path = path
        self.dismiss_popup()


class Editor(App):
    pass


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)


if __name__ == '__main__':
    Editor().run()