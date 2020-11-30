import pickle

import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from os import listdir
from Skeltonizer_GCP import ImageDataSet, Skeltonizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from JustF1Score import compute_F1

"""
This program loads a trained model to get results from. Data is loaded from "data.pickle" with keys:
    'train_in', 'train_target', 'val_in', 'val_target', 'test_in', 'test_target'

It has two main  purposes:
    - Generate results for a given threshold
    - Generate Plot for different thresholds

Optional experiments can be added as functions to be computed by necessary flags

"""

def compute(model, model_name, data_loader, thresh =0.8):
    """
    compute outputs given dataset and threshold
    return result array
    """

    count = 0
    result_array = np.empty(shape=(len(data_loader), 256, 256))
    for data, target in data_loader:
        input = data.to(device)
        target = target.to(device)

        output = model(input)

        # output result as an image
        p = output.cpu().detach().numpy()
        t = target.cpu().squeeze()
        i = input.cpu().squeeze()
        p[p > thresh] = 255
        p[p <= thresh] = 0
        result_array[count, ...] = p.squeeze()

        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(p.squeeze(), cmap='gray')
        axarr[0].set_title('Prediction')
        axarr[1].imshow(t, cmap='gray')
        axarr[1].set_title('Target')
        axarr[2].imshow(i, cmap='gray')
        axarr[2].set_title('Input')
        plt.savefig('./results/'+ model_name + '_thresh'+str(thresh)+'_pred'+ str(count) + '.png')
        plt.close()
        #plt.show()
        count += 1
        # p[torch.argmax(output, keepdim=True)] = 255
        # p[output<=0.9] = 0
        # result = (p).int()
        # trans = transforms.ToPILImage()
        # image = trans(result[0])
        # image.save("output_result.png", "PNG")
        # image.show()
        # time.sleep(1)
    # print(result_array.shape)
    #np.save("result", result_array)
    print("Threshold: " + str(thresh))

    return result_array

def threshold_study(model, model_name, t_start, t_end, step, data_loader, targets):
    """
    Compute F1 score for a range of thresholds.
    save figure for the curve of F1 score
    """
    #thresh_study
    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    thresholds = np.arange(t_start, t_end, step)
    F1_scores = np.empty_like(thresholds)
    #targets = data_loader
    for i, thresh in enumerate(thresholds):
        result = compute(model, model_name, data_loader, thresh)
        F1_scores[i] = compute_F1(result, targets)


    np.save("threshold_study_"+model_name, np.array([thresholds, F1_scores]))


def main(args):
    """
    Perform the experiments that are flagged.
    """
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run experiments on trained model.")
    parser.add_argument('-m', '--model', help='File path to the model')
    parser.add_argument('thresholds', metavar='T', type=float, nargs='?')

    args = parser.parse_args()
    num_workers=0
    batch_size = 1  # batch size

    #Load the data
    print("Loading Data...")
    data = pickle.load(open("data.pickle", "rb"))
    test_dataset = TensorDataset(torch.from_numpy(data["test_in"]).float(), torch.from_numpy(data["test_target"]).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_size = 1.0 * len(test_loader)
    print("Done.")

    #Load the model
    s= args.model #'./models/models_e24_loss0.012'

    model_name = s.replace('./models/', '')
    model = Skeltonizer()
    model.load_state_dict(torch.load(s))
    model.eval()
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #run whatever experiment we want.
    t_start = 0.6
    t_end = 1.0
    t_step = 0.01
    threshold_study(model, model_name, t_start, t_end, t_step, test_loader, data['test_target'].squeeze() * 255.0)
