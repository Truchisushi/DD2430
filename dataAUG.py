import re
import PIL
from PIL import Image
from os import listdir
from os.path import isfile, join
from matplotlib import image
from matplotlib import pyplot as plt
from numpy import asarray
from PIL import ImageOps
import numpy as np
import csv
import pickle

xs = [ './data/img_train_shape/'+ f for f in listdir('./data/img_train_shape/')]
ys = [ './data/img_train2/'+ f for f in listdir('./data/img_train2/')]


i = 0
for f in xs:
    try:
        pic = Image.open(f)
        #im = ImageOps.fit(pic,(256,256),method=3, bleed=0.0, centering=(0.5, 0.5))
        imArray1 = np.array(pic)

        im = pic.rotate(90)
        imArray2 = np.array(im)

        im = pic.rotate(180)
        imArray3 = np.array(im)

        im = pic.rotate(270)
        imArray4 = np.array(im)

        im = ImageOps.mirror(pic)
        imArray5 = np.array(im)

        im = ImageOps.mirror(pic).rotate(90)
        imArray6 = np.array(im)

        im = ImageOps.mirror(pic).rotate(180)
        imArray7 = np.array(im)

        im = ImageOps.mirror(pic).rotate(270)
        imArray8 = np.array(im)


    except Exception:
        print('error image')
        pass
    else:
            print('done '+f)

            i = i+1
            s = 'data/img_train_shape_AUG/' + str(i) + '_xxxx.png'
            s1 = s.replace('xxxx','1')
            s2 = s.replace('xxxx','2')
            s3 = s.replace('xxxx','3')
            s4 = s.replace('xxxx','4')
            s5 = s.replace('xxxx','5')
            s6 = s.replace('xxxx','6')
            s7 = s.replace('xxxx','7')
            s8 = s.replace('xxxx','8')


            Image.fromarray(imArray1).save(s1)
            Image.fromarray(imArray2).save(s2)
            Image.fromarray(imArray3).save(s3)
            Image.fromarray(imArray4).save(s4)
            Image.fromarray(imArray5).save(s5)
            Image.fromarray(imArray6).save(s6)
            Image.fromarray(imArray7).save(s7)
            Image.fromarray(imArray8).save(s8)

i = 0
for f in ys:
    try:
        pic = Image.open(f)
        #im = ImageOps.fit(pic,(256,256),method=3, bleed=0.0, centering=(0.5, 0.5))
        imArray1 = np.array(pic)

        im = pic.rotate(90)
        imArray2 = np.array(im)

        im = pic.rotate(180)
        imArray3 = np.array(im)

        im = pic.rotate(270)
        imArray4 = np.array(im)

        im = ImageOps.mirror(pic)
        imArray5 = np.array(im)

        im = ImageOps.mirror(pic).rotate(90)
        imArray6 = np.array(im)

        im = ImageOps.mirror(pic).rotate(180)
        imArray7 = np.array(im)

        im = ImageOps.mirror(pic).rotate(270)
        imArray8 = np.array(im)


    except Exception:
        print('error image')
        pass
    else:
            print('done '+f)

            i = i+1
            s = 'data/img_train2_AUG/' + str(i) + '_xxxx.png'
            s1 = s.replace('xxxx','1')
            s2 = s.replace('xxxx','2')
            s3 = s.replace('xxxx','3')
            s4 = s.replace('xxxx','4')
            s5 = s.replace('xxxx','5')
            s6 = s.replace('xxxx','6')
            s7 = s.replace('xxxx','7')
            s8 = s.replace('xxxx','8')


            Image.fromarray(imArray1).save(s1)
            Image.fromarray(imArray2).save(s2)
            Image.fromarray(imArray3).save(s3)
            Image.fromarray(imArray4).save(s4)
            Image.fromarray(imArray5).save(s5)
            Image.fromarray(imArray6).save(s6)
            Image.fromarray(imArray7).save(s7)
            Image.fromarray(imArray8).save(s8)
