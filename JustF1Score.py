import numpy as np
from os import listdir
from os.path import isfile, join
import copy
import math
from PIL import Image

if __name__ == "__main__":

        ys = [ './data/validation_output/'+ f for f in listdir('./data/validation_output/')]
        results = np.load("result_120x256x256.npy")
        f1_score = 0
        for i in range(120):
            validations = (np.asarray(Image.open(ys[i]))).astype(np.float32)
            trueNegatives = 0
            falseNegatives = 0
            truePositives = 0
            falsePositives = 0
            for r in range(256):
                for c in range(256):
                    if(results[i][r][c] == 0.0):
                        if(np.any(validations[r][c] == 0.0)):
                            trueNegatives = trueNegatives + 1
                        else:
                            falseNegatives = falseNegatives + 1

                    if(results[i][r][c] == 255.0):
                        if(np.any(validations[r][c] == 255.0)):
                            truePositives = truePositives + 1
                        else:
                            falsePositives = falsePositives + 1

            precision = truePositives / (truePositives + falsePositives)
            recall = truePositives / (truePositives + falseNegatives)

            f1_score = f1_score + (2*precision*recall / (precision + recall) / 120)

        print(f1_score)
