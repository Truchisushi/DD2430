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

class Skeltonizer(nn.Module):
    def __init__(self):

        super(Skeltonizer, self).__init__()
        #Initiate all layers
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Downsamp1 = nn.Conv2d(1,32,kernel_size=3,padding=(1,1))
        self.DownConv1 = self.DownConv(32)
        self.Downsamp2 = nn.Conv2d(32,64,kernel_size=3,padding=(1,1))
        self.DownConv2 = self.DownConv(64)
        self.Downsamp3 = nn.Conv2d(64,128,kernel_size=3,padding=(1,1))
        self.DownConv3 = self.DownConv(128)
        self.Downsamp4 = nn.Conv2d(128,256,kernel_size=3,padding=(1,1))
        self.DownConv4 = self.DownConv(256)
        self.Downsamp5 = nn.Conv2d(256,512,kernel_size=3,padding=(1,1))
        self.DownConv5 = self.DownConv(512)
        self.BottleNeck1 = nn.Conv2d(512,1024,kernel_size=1)
        self.ReLU = nn.ReLU(inplace = True)
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

    def DownConv(self,output_size):
        dnc = nn.Sequential(
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=(1,1)),
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=(1,1)),
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=(1,1)),
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=(1,1)),
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=(1,1))
        )
        return dnc

    def UpConv(self,input_size, output_size):
        upc = nn.Sequential(
            nn.Conv2d(input_size,output_size,kernel_size = 3,padding=(1,1)), # *2 Denote convolution layer
            nn.Conv2d(output_size,output_size,kernel_size = 3,padding=(1,1)),
            nn.Conv2d(output_size,output_size,kernel_size = 3,padding=(1,1)),
            nn.Conv2d(output_size,output_size,kernel_size = 3,padding=(1,1))
        )
        return upc

    def forward(self, x):
        #Expand feature space for input image
        #x = self.inputConv(x)
        #Decoder
        x1 = self.Downsamp1(x)
        x1 = self.DownConv1(x1) + x1
        x2 = self.MaxPool(x1)

        x3 = self.Downsamp2(x2)
        x3 = self.DownConv2(x3) + x3
        x4 = self.MaxPool(x3)

        x5 = self.Downsamp3(x4)
        x5 = self.DownConv3(x5) + x5
        x6 = self.MaxPool(x5)

        x7 = self.Downsamp4(x6)
        x7 = self.DownConv4(x7) + x7
        x8 = self.MaxPool(x7)

        x9 = self.Downsamp5(x8)
        x9 = self.DownConv5(x9) + x9
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
    image = loader(image).unsqueeze(0)
    return image


def my_loss(output, target):
    ##################### Cross entropy & Dice Loss ##############################
    ##############Edit: Weighted Cross entropy Loss balanced with Dice Loss Loss ##################
    #Balancing weight for loss functions
    w1 = 0.01
    w2 = 100
    #Weight for cross entropy loss
    wpos = 50
    wneg = 0.75
    eps = np.finfo(float).eps
    diceLoss = (1 - (2*(torch.mul(target, output).sum())+eps) / ((target.sum() + output.sum()+eps)))

    #To avoid nan
    logo1 = torch.log(eps+output)
    L1 = torch.mul(target,logo1)
    L1[target == 0] = 0

    l2 = 1.0 - output
    logo2 = torch.log(eps+l2)
    L2 = torch.mul(1-target,logo2)
    L2[target == 1] = 0

    L = - (wpos*L1+wneg*L2).sum()
    #print(w1*L)
    #print(w2*diceLoss)
    Loss = w1*L + w2*diceLoss
    #print(Loss)
    return Loss

if __name__ == "__main__":

        # Save data
        xs = [ './data/img_train_shape/'+ f for f in listdir('./data/img_train_shape/')]
        ys = [ './data/img_train2/'+ f for f in listdir('./data/img_train2/')]
        batchSize = len(xs)

        #create model, or load model
        model = Skeltonizer()
        #model.load_state_dict(torch.load('./models'))
        #model.eval()

        model.cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # 500 epochs
        for i in range(500):
            # 1218 samples for training
            for j in range(1118):
                input = image_loader(xs[j])
                input = input.to(device)
                target = image_loader(ys[j])
                target = target.to(device)

                optimizer.zero_grad()   # zero the gradient buffers
                output = model(input)
                loss = my_loss(output,target)
                loss.backward()
                optimizer.step()

        #save the model after the training
        torch.save(model.state_dict(), './models')

        #output result as an image
        p = output.cpu()
        p[output>0.9] = 255
        p[output<=0.9] = 0
        result = (p).int()
        trans = transforms.ToPILImage()
        image = trans(result[0])
        image.show()
