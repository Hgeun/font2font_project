import matplotlib.pyplot as plt
import numpy as np
import torch

import PIL.Image

from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

#network_para = 'best_point_1222_20_00_16.pth'

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
#net.load_state_dict(torch.load(network_para))

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

train_list = np.genfromtxt('./fontConvNet-master/trainset.txt',dtype=None,encoding='UTF-8')

trans = transforms.ToTensor()

min_loss = 1
for epoch in range(2000):  # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    running_loss_epoch = 0.0
    for i,(input_path,target_path) in enumerate(train_list):
        input = PIL.Image.open(input_path)#.convert('L')
        target = PIL.Image.open(target_path)#.convert('L')
        input=trans(input).reshape(1,3,64,64)

        target=trans(target).reshape(1,3,64,64)
        optimizer.zero_grad()
        input, target = input.to(device), target.to(device)
        output = net(input)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_epoch += loss.item()
        if i % 100 == 99:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        if i == len(train_list)-1 :
            epoch_loss = running_loss_epoch/len(train_list)
            running_loss_epoch = 0.0
            if min_loss > epoch_loss:
                min_loss = epoch_loss
                print('min loss : %.3f' % (epoch_loss))
                torch.save(net.state_dict(), './ver2_best_point.pth')

torch.save(net.state_dict(),'./ver2_final_point_' + datetime.today().strftime("%m%d_%H_%M_%S") +'.pth')
print('Finished Training')




