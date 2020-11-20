import numpy as np
import torch
from matplotlib import pyplot as plt
from os import listdir
from Skeltonizer_GCP import ImageDataSet, Skeltonizer
from torch.utils.data import DataLoader, Dataset
from JustF1Score import main

batch_size = 1  # batch size
shuffle = False  # data augmentation shuffling
epochs = 500  # number of epochs
num_workers = 0


# Save data
xs = ['./data/img_train_shape/' + f for f in listdir('./data/img_train_shape/')]
ys = ['./data/img_train2/' + f for f in listdir('./data/img_train2/')]

xs_val = ['./data/validation_input/' + f for f in listdir('./data/validation_input/')]
ys_val = ['./data/validation_output/' + f for f in listdir('./data/validation_output/')]

train_data = ImageDataSet(xs, ys)
val_data = ImageDataSet(xs_val, ys_val)

print("Configuring DataLoader for training set")
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
train_size = len(train_loader)
print("Done.")

print("Configuring DataLoader for Validation set")
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
val_size = len(val_loader)
print("Done.")

# create model, or load model
s = 'focal_loss_expLRdecay096/models_e35'


model = Skeltonizer()
model.load_state_dict(torch.load(s))
model.eval()
#print(model)

#print(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for thresh in np.arange(0.6, 1.0, 0.01):
    count = 0
    result_array = np.empty(shape=(val_size, 256, 256))
    for data, target in val_loader:
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
        axarr[1].imshow(t, cmap='gray')
        axarr[2].imshow(i, cmap='gray')
        #plt.savefig('train' + str(count) + '.png')
        plt.show()
        count += 1
        # p[torch.argmax(output, keepdim=True)] = 255
        # p[output<=0.9] = 0
        # result = (p).int()
        # trans = transforms.ToPILImage()
        # image = trans(result[0])
        # image.save("output_result.png", "PNG")
        # image.show()
        # time.sleep(1)
    #print(result_array.shape)
    np.save("result", result_array)
    print("Threshold: " + str(thresh))
    main()
