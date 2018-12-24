import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import PIL.Image

from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

network_para = 'ver2_best_point.pth'
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 9, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(9)
        self.conv3 = nn.Conv2d(9, 15, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(15)
        self.conv4 = nn.Conv2d(15, 5, 5, 1, 2)
        self.bn4 = nn.BatchNorm2d(5)
        self.conv5 = nn.Conv2d(5, 3, 5, 1, 2)
        self.bn5 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x
''' # prev network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 9, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(9)
        self.conv3 = nn.Conv2d(9, 16, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 16, 5, 1, 2)
        self.bn8 = nn.BatchNorm2d(16)
        self.conv9 = nn.Conv2d(16, 8, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(8)
        self.conv10 = nn.Conv2d(8, 3, 5, 1, 2)
        self.bn10 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        return x

net = Net()
net.load_state_dict(torch.load(network_para))
test_list = np.genfromtxt('./fontConvNet-master/testset.txt',dtype=None,encoding='UTF-8')

criterion = nn.L1Loss()

trans = transforms.ToTensor()
trans_img = transforms.ToPILImage()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

try:
    if not(os.path.isdir('./ver2_output_'+ datetime.today().strftime("%m%d_%H_%M")+'/')):
        os.makedirs(os.path.join('./ver2_output_'+ datetime.today().strftime("%m%d_%H_%M")+'/'))
except OSError as e:
    if e.errno != e.errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise

with torch.no_grad():
    running_loss = 0.0
    for i,(input_path,target_path) in enumerate(test_list):
        input = PIL.Image.open(input_path)
        target = PIL.Image.open(target_path)
        input = trans(input).reshape(1, 3, 64, 64)
        target = trans(target).reshape(1, 3, 64, 64)
        input, target = input.to(device), target.to(device)

        output = net(input)
        loss = criterion(output, target)
        running_loss += loss.item()
        if i == len(test_list)-1:  # print every 2000 mini-batches
            print('[%5d] loss: %.3f' %
                  (i + 1, running_loss / len(test_list)))
            running_loss = 0.0


        output1 = (output.data).cpu()  # output1 tensor
        #print(output1.shape)
        output1 = (output1.reshape(3, 64, 64) * 255).numpy()
        #print(output1.shape)
        output1[output1>255] = 255
        output1 = output1.astype(np.uint8)
        #print(output1.shape)
        mean1 = np.mean(output1, axis=0)
        #mean1[mean1>255]=255
        plt.imsave('./ver2_output_'+ datetime.today().strftime("%m%d_%H_%M")+'/'+input_path[-9:],mean1,cmap=plt.get_cmap('gray'))

print('Finished Testing')