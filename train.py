import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt

import time
import argparse

import numpy as np

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader

from Skeltonizer_GCP import Skeltonizer, my_loss


def main(args):
    """
    Main part of the training. arguments sent in to main() to define training. Arguments could include:
        - Epochs
        - Batch size
        - Verbosity
    :return:
    """
    epochs = args.epochs
    batch_size = args.batchsize
    gamma = args.gamma
    learning_rate = args.LR
    print("Epochs: " + str(epochs) + ", Batch size: " + str(batch_size) + ", LR_decay: " + str(gamma) +", LR: "+ str(learning_rate))
    num_workers = args.workers    #Change this to appropriate number of workers

    # -------------------------------Data Loader--------------------------------------------

    # Load Data
    print("Loading Data...")
    data = pickle.load(open("data.pickle", "rb"))
    train_dataset = TensorDataset(torch.from_numpy(data["train_in"]).float(), torch.from_numpy(data["train_target"]).float())
    val_dataset = TensorDataset(torch.from_numpy(data["val_in"]).float(), torch.from_numpy(data["val_target"]).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_size = 1.0 * len(train_loader)
    val_size = 1.0 * len(val_loader)
    print("Done.")

    # -----------------------------------create model, or load model------------------------------------
    model = Skeltonizer()
    # model.load_state_dict(torch.load('./models'))
    model.eval()
    print(model)

    # --------------------------------------------------------------------------------------------------
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")  # Make sure GPU is compatible with CUDA
    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    e = np.arange(0, epochs)
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    # Adjust learning rate by multiplying LR-factor with lambda:

    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # ---------------------------------------------------train--------------------------------------------
    model.train()
    best_val = np.finfo(float).max
    for i in range(epochs):
        start = time.time()
        count = 0
        # samples for training
        #train_loss = 0.0
        #val_loss = 0.0
        for data, target in train_loader:
            print('Epoch %d: %d/%d' % (i, count, train_size), end='\r')
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()  # zero the gradient buffers
            output = model(data)
            loss = my_loss(output, target)
            loss.backward()
            optimizer.step()

            train_losses[i] += loss.item() / train_size

            count += 1
        print("")
        # model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = my_loss(output, target)
                val_losses[i] += loss.item() / val_size

        end = time.time()
        lr = 0
        for param_group in optimizer.param_groups: lr = param_group['lr']
        print("Epoch %d/%d, %d s, loss: %f, val: %f, lr: %f" % (
        i, epochs, end - start, train_losses[i], val_losses[i], lr))

        scheduler.step()

        if (val_losses[i] < best_val):
            print("Saving model!")
            best_val = val_losses[i]
            torch.save(model.state_dict(), './models/models_e' + str(i) + '_loss' + str(np.round(val_losses[i], 4)))
        elif (i % 5 == 0): #save model when val_loss is improved, or every other epoch
            print("Saving model!")
            torch.save(model.state_dict(), './models/models_e' + str(i) + '_loss' + str(np.round(val_losses[i], 4)))
    # save the model after the training
    # torch.save(model.state_dict(), './models')

    plt.plot(e, train_losses)
    plt.plot(e, val_losses)
    plt.legend(["train", "val"])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("loss_plot.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process to train U-net Skeltonizer. When running into memory issues, try lowering batchsize and(or) workers')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('-b', '--batchsize', type=int, help='Mini-batch size', default=16)
    parser.add_argument('-g', '--gamma', type=float, help='Exponential Learning Rate Decay', default=0.96)
    parser.add_argument('-l', '--LR', type=float, help='Learning Rate', default=0.0001)
    parser.add_argument('-w', '--workers', type=int, help='Amount of concurrent workers.', default=4)

    main(parser.parse_args())












