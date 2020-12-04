import numpy as np
from os import listdir
from os.path import isfile, join
import copy
import math
from PIL import Image

if __name__ == "__main__":

        xs = [ './data/img_train_shape/'+ f for f in listdir('./data/img_train_shape/')]
        ys = [ './data/img_train2/'+ f for f in listdir('./data/img_train2/')]


        y = np.array(Image.open(ys[45]))
        x = np.array(Image.open(xs[45]))

        for r in range(256):
            for c in range(256):
                z = y[r][c][0].astype(int)
                if(z > 1):
                    x[r][c] = 0

        img = Image.fromarray(x)
        img.show()

