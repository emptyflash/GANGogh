"""
A script designed to 1) resize all of the downloaded images to desired dimension (DEFAULT 64x64 pixels) and 2) rename images in folders from 1.png to n.png for ease of use in training
"""

import os
import scipy.misc
import random

root='/media/cameron/Keys/wikiart'


#Set your own PATH 
PATH = os.path.normpath('/home/cameron/Projects/GANGogh/wikiart_small/')
generator = os.walk(root)
next(generator)

for subdir, dirs, files in generator:
    style = os.path.relpath(subdir, root)
    name =  style
    if len(style) < 1:
        continue
    try:
        os.stat(PATH + "/" + name)
    except:
        os.mkdir(PATH + "/" + name)
    
    for i, f in enumerate(files):
        source = root + '/' + style + '/' + f
        print(i, source)
        try:
            image = scipy.misc.imread(source)
            image = scipy.misc.imresize(image,(128,128))
            scipy.misc.imsave(PATH + "/" + name + '/' + str(i) + '.png',image)
        except Exception as e:
            print(e)
            print('missed it: ' + source)
