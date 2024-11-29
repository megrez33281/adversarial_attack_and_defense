import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  # neural network model
  def __init__(self):
    super(Net, self).__init__()
    self.PictureSize = 28
    # Use FashionMNIST dataset
    # picture size 28*28
    # Input 1 channel, Output 32 channel, kernel=3, stride=1（步幅）
    # for every channel：
    # 28*28 -> ((28-kernel size)/stride + 1)* ((28-kernel size)/stride + 1) -> 26*26
    # Thus, after conv1, there are 32 channels, every channel has 26*26 matrix
    self.conv1 = nn.Conv2d(1, 32, 3, 1)

    # Input 32 channel, Output 64 channel, kernel=3, stride=1（步幅）
    # for every channel：
    # 26*26 -> ((26-kernel size)/stride + 1)* ((26-kernel size)/stride + 1) -> 24*24
    # Thus, after conv2, there are 64 channels, every channel has 24*24 matrix
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    # set 0.25 of neuron to 0
    self.dropout1 = nn.Dropout(0.25)
    # set 0.5 of neuron to 0
    self.dropout2 = nn.Dropout(0.5)
    # flatten（展平） after pool layer
    # because the pool size is 2*2
    # after pool layer, the matrix size = (24/2)*(24/2) = 12*12 for every channel
    # when flatten the feature to linear, we get 64*12*12 = 9216
    # fc1（全連接層1）, input size: 1*9216, output size: 1*128
    self.PictureSize -= 2*2 # two conv layer with kernel=3 stride = 1
    self.PictureSize = int(self.PictureSize/2) # pool layer 2*2 
    self.fc1 = nn.Linear(64*self.PictureSize*self.PictureSize , 128)
    # fc2（全連接層2）, input size: 1*128, output size: 1*10
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output
  
class NetF(nn.Module):
    def __init__(self):
        # for FashionMNIST
        super(NetF, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class NetF1(nn.Module):
    def __init__(self):
        super(NetF1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x