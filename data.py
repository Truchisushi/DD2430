
import pickle
from os import listdir

import numpy as np

import argparse

from PIL import Image, ImageOps
from matplotlib import image
import matplotlib.pyplot as plt
from torchvision import transforms

"""
read data from folders:
    Starting Kit Pixel/img_train_shape
    Starting Ket Pixel/img_train
    
    or from specific folder with already predetermined validation set.

Augment the data: True

pickle the files with keys: "shape_train", "skeleton_train" & "shape_val", "skeleton_val"

Be sure to remove .DS_Store


 """

def augment_data(f, imsize):
    """
    Augm
    :param f: file path
    :return: return numpy list of augmented and normalized images
    """
    #print(f)
    pic = Image.open(f).convert('F')  #Open
    #w, h = pic.size

    imgs_array = np.empty((8, imsize, imsize))

    imgs_array[0, ...] = np.array(pic)

    im = pic.rotate(90)
    imgs_array[1, ...] = np.array(im)

    im = pic.rotate(180)
    imgs_array[2, ...] = np.array(im)

    im = pic.rotate(270)
    imgs_array[3, ...] = np.array(im)

    im = ImageOps.mirror(pic)
    imgs_array[4, ...] = np.array(im)

    im = ImageOps.mirror(pic).rotate(90)
    imgs_array[5, ...] = np.array(im)

    im = ImageOps.mirror(pic).rotate(180)
    imgs_array[6, ...] = np.array(im)

    im = ImageOps.mirror(pic).rotate(270)
    imgs_array[7, ...] = np.array(im)


    return imgs_array / 255.0   #Normalize

def main(args):
    imsize = 256

    use_testset = args.testset

    #These are to generate random datasets.
    dir_in = './Starting Kit Pixel/img_train_shape'
    dir_target = './Starting Kit Pixel/img_train2'
    test_dir_in = './data/validation_input'
    test_dir_target = './data/validation_output'

    x, y, x_test, y_test = np.array([]), np.array([]), np.array([]), np.array([])

    if use_testset:
        dir_in = './data/img_train_shape'
        dir_target = './data/img_train2'




    data_target = 'data.pickle'

    #Create numpy array for input and targets and augment them
    x = np.array([augment_data(dir_in + '/' + img, imsize) for img in listdir(dir_in)]).reshape(-1, 1, imsize, imsize).astype(float)
    y = np.array([augment_data(dir_target + '/' + img, imsize) for img in listdir(dir_target)]).reshape(-1, 1, imsize, imsize).astype(float)

    if use_testset:
        x_test = np.array([augment_data(test_dir_in + '/' + img, imsize) for img in listdir(test_dir_in)]).reshape(-1, 1, imsize, imsize).astype(float)
        y_test = np.array([ augment_data(test_dir_target + '/' + img, imsize)  for img in listdir(test_dir_target)]).reshape(-1, 1, imsize, imsize).astype(float)

    #shuffle:
    indices = np.random.permutation(x.shape[0])
    x = x[indices, ...]
    y = y[indices, ...]



    #print("Input data shape:", x.shape)
    #print("Output data shape:", y.shape)


    for i in range(24):
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(x[i].squeeze(), cmap='gray')
        axarr[1].imshow(y[i].squeeze(), cmap='gray')
        # plt.savefig('train' + str(count) + '.png')
        plt.show()


    #Time to split into training, validation and test sets and pickle

    split = lambda a: np.split(a, [int(len(a) * 0.8), int(len(a) * 0.9), len(a)], axis=0)

    if use_testset:
        split = lambda a: np.split(a, [int(len(a) * 0.8), len(a)], axis=0)

    d = {}
    #tmp = split(x[indices, ...])
    if use_testset:
        d['train_in'], d['val_in'], _ = split(x[indices, ...])
        d['train_target'], d['val_target'], _ = split(y[indices, ...])
        d['test_in'] = x_test
        d['test_target'] = y_test

    else:
        d['train_in'], d['val_in'], d['test_in'], _ = split(x[indices, ...])
        d['train_target'], d['val_target'], d['test_target'], _ = split(y[indices, ...])

    pickle.dump(d, open(data_target, "wb"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate data.")
    parser.add_argument('-t', '--testset', type=bool, help='Use a predifned testset at "data/validation_input_AUG" & "data/validation_output_AUG"', default=True)
    args = parser.parse_args()
    main(args)



