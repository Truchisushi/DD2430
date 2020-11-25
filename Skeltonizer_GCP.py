import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from os import listdir
from os.path import isfile, join
import copy
import math
import time
from scipy.ndimage import distance_transform_edt as distance_trans

import scipy.misc
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.data import DataLoader, Dataset


class Skeltonizer(nn.Module):
    def __init__(self):

        super(Skeltonizer, self).__init__()
        #Initiate all layers
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.DownConv1 = self.DownConv(1, 32)
        self.DownConv2 = self.DownConv(32, 64)
        self.DownConv3 = self.DownConv(64, 128)
        self.DownConv4 = self.DownConv(128, 256)
        self.DownConv5 = self.DownConv(256, 512)
        self.BottleNeck1 = nn.Conv2d(512,1024,kernel_size=1)
        self.ReLU = nn.ReLU()
        self.BottleNeck2 = nn.Conv2d(1024,1024,kernel_size=1)
        self.convT1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.UpConv1 = self.UpConv(1024,512)
        self.convT2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.UpConv2 = self.UpConv(512,256)
        self.convT3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.UpConv3 = self.UpConv(256,128)
        self.convT4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.UpConv4 = self.UpConv(128,64)
        self.convT5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.UpConv5 = self.UpConv(64,32)
        self.Outputlayer = nn.Conv2d(32,1,kernel_size=1)

    def DownConv(self,input_size, output_size):
        dnc = nn.Sequential(
            nn.Conv2d(input_size,output_size,kernel_size=3,padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=(1,1))
        )
        return dnc

    def UpConv(self,input_size, output_size):
        upc = nn.Sequential(
            nn.Conv2d(input_size,output_size,kernel_size = 3,padding=(1,1)), # *2 Denote convolution layer
            nn.ReLU(),
            nn.Conv2d(output_size,output_size,kernel_size = 3,padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(output_size,output_size,kernel_size = 3,padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(output_size,output_size,kernel_size = 3,padding=(1,1))
        )
        return upc

    def forward(self, x):
        #apply distance transform:




        #Expand feature space for input image
        #x = self.inputConv(x)
        #Decoder
        #x1 = self.Downsamp1(x)
        #x1 = self.DownConv1(x1) + x1
        x1 = self.DownConv1(x)
        #x1 = self.Downsamp1(x1)
        x2 = self.MaxPool(x1)


        #x3 = self.Downsamp2(x2)
        #x3 = self.DownConv2(x3) + x3
        x3 = self.DownConv2(x2)
        #x3 = self.Downsamp2
        x4 = self.MaxPool(x3)

        #x5 = self.Downsamp3(x4)
        #x5 = self.DownConv3(x5) + x5
        x5 = self.DownConv3(x4)
        #x5 = self.Downsamp3(x5)
        x6 = self.MaxPool(x5)

        #x7 = self.Downsamp4(x6)
        #x7 = self.DownConv4(x7) + x7
        x7 = self.DownConv4(x6)
        #x7 = self.Downsamp4(x7)
        x8 = self.MaxPool(x7)

        #x9 = self.Downsamp5(x8)
        #x9 = self.DownConv5(x9) + x9
        x9 = self.DownConv5(x8)
        #x9 = self.Downsamp5(x9)
        x10 = self.MaxPool(x9)

        #Bottle Neck
        x11 = self.BottleNeck1(x10)
        x12 = self.ReLU(x11)
        x13 = self.BottleNeck2(x12)
        x14 = self.ReLU(x13)

        #Encoder
        x_temp = self.convT1(x14)
        x15 = self.UpConv1(torch.cat([x_temp,x9],1))
        x_temp = self.convT2(x15)
        x16 = self.UpConv2(torch.cat([x_temp,x7],1))
        x_temp = self.convT3(x16)
        x17 = self.UpConv3(torch.cat([x_temp,x5],1))
        x_temp = self.convT4(x17)
        x18 = self.UpConv4(torch.cat([x_temp,x3],1))
        x_temp = self.convT5(x18)
        x19 = self.UpConv5(torch.cat([x_temp,x1],1))
        output = torch.sigmoid(self.Outputlayer(x19))

        return output


imsize = 256
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name).convert('1')
    image = loader(image) #.unsqueeze(0)
    return image

def load_data(xs, ys):
    """

    :param xs: input
    :param ys: test
    :return:   data - array with all preloaded data
    """
    return list(zip(xs, ys))

class ImageDataSet(Dataset):

    def __init__(self, images, targets, dataset_size=-1):
        self.size = len(images) if dataset_size == -1 else dataset_size
        self.images = torch.empty(self.size, 1, imsize, imsize)
        self.targets = torch.empty(self.size, 1, imsize, imsize)
        for i in range(self.size):
            #tmp = distance_trans(image_loader(images[i]).numpy())    #Apply distance transform
            self.images[i, ...] = image_loader(images[i]) #torch.from_numpy(tmp / tmp.max() )     #Normalize
            self.targets[i, ...] = image_loader(targets[i])

        #self.images = images[:dataset_size]
        #self.targets = targets[:dataset_size]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        #X = image_loader(self.images[index])
        #y = image_loader(self.targets[index])
        X = self.images[index, ...]
        tmp = X.numpy()
        y = self.targets[index, ...]

        return X, y

def my_loss(output, target):
   
    ################# Weighted Focal loss + dice loss #####################

    #Balancing weight for loss functions
    wpos = 50
    wneg = 0.75
    gamma = 2

    logo1 = torch.log(eps+output)
    L1 = torch.mul(target,logo1)
    pos_focal = (1-output)**gamma
    L1 = torch.mul(pos_focal,L1)
    L1[target == 0] = 0

    l2 = 1.0 - output
    logo2 = torch.log(eps+l2)

    L2 = torch.mul(1-target,logo2)
    neg_focal = (1-l2)**gamma
    L2 = torch.mul(neg_focal,L2)
    L2[target == 1] = 0

    Loss = -(wpos * L1 + wneg * L2).sum()/ (output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3])
    return Loss

if __name__ == "__main__":
        batch_size = 16      #batch size.
        shuffle = True     #data augmentation shuffling. Set to true to shuffle
        epochs = 100       #number of epochs
        num_workers = 4
        dataset_size = -1   #Change this to the number of images to test on


        #-------------------------------Data Loader--------------------------------------------
        
        xs = [ './data/img_train_shape_AUG/'+ f for f in listdir('./data/img_train_shape_AUG/')]
        ys = [ './data/img_train2_AUG/'+ f for f in listdir('./data/img_train2_AUG/')]


        xs_val = ['./data/validation_input/'+ f for f in listdir('./data/validation_input/')]
        ys_val = [ './data/validation_output/'+ f for f in listdir('./data/validation_output/')]

        train_data = ImageDataSet(xs, ys, dataset_size)
        val_data = ImageDataSet(xs_val, ys_val)

        print("Configuring DataLoader for training set")
        train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        train_size = 1.0 * len(train_loader)
        print("Done.")

        print("Configuring DataLoader for Validation set")
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        val_size = 1.0 * len(val_loader)
        print("Done.")

        
        #-----------------------------------create model, or load model------------------------------------
        model = Skeltonizer()
        #model.load_state_dict(torch.load('./models'))
        model.eval()
        print(model)
        
        #--------------------------------------------------------------------------------------------------
        print(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu") #Make sure GPU is compatible with CUDA
        model.cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)


        e = np.arange(0, epochs)
        train_losses = np.zeroes(epochs)
        val_losses = np.zeroes(epochs)

        #Adjust learning rate by multiplying LR-factor with lambda:
        gamma = 0.95 #lambda lmbda: 0.95
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        #---------------------------------------------------train--------------------------------------------
        model.train()
        for i in range(epochs):
            start = time.time()
            count = 0
            # samples for training
            train_loss = 0.0
            val_loss = 0.0
            for data, target in train_loader:
                print(count)
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()   # zero the gradient buffers
                output = model(data)
                loss = my_loss(output, target)
                loss.backward()
                optimizer.step()

                train_losses[i] += loss.item()/train_size

            #model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    loss = my_loss(output, target)
                    val_losses[i] += loss.item()/val_size

            end = time.time()
            lr = 0
            for param_group in optimizer.param_groups: lr=param_group['lr']
            print("Epoch %d/%d, %d s, loss: %f, val: %f, lr: %f" % (i, epochs, end - start, train_losses[i], val_losses[i], lr))

            scheduler.step()

            if (i % 5 == 0):
                print("Saving model!")
                torch.save(model.state_dict(), './models_e'+str(i))
        # save the model after the training
        torch.save(model.state_dict(), './models')

        plt.plot(e, train_losses)
        plt.plot(e, val_losses)
        plt.legend(["train", "val"])
        plt.savefig("loss_plot.png")
        plt.show()
