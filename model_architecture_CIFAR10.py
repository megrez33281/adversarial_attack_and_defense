import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  # neural network model
  def __init__(self):
    super(Net, self).__init__()
    self.PictureSize = 32
    # Use FashionMNIST dataset
    # picture size 32*32
    # Input 1 channel, Output 64 channel, kernel=3, stride=1（步幅）
    # for every channel：
    # 32*32 -> ((32-kernel size)/stride + 1)* ((32-kernel size)/stride + 1) -> 30*30
    # Thus, after conv1, there are 64 channels, every channel has 30*30 matrix
    self.conv1 = nn.Conv2d(3, 64, 3, 1)

    # Input 64 channel, Output 128 channel, kernel=3, stride=1（步幅）
    # for every channel：
    # 30*30 -> ((30-kernel size)/stride + 1)* ((30-kernel size)/stride + 1) -> 28*28
    # Thus, after conv2, there are 128 channels, every channel has 28*28 matrix
    self.conv2 = nn.Conv2d(64, 128, 3, 1)
    # set 0.25 of neuron to 0
    self.dropout1 = nn.Dropout(0.25)
    # set 0.5 of neuron to 0
    self.dropout2 = nn.Dropout(0.5)
    # flatten（展平） after pool layer
    # because the pool size is 2*2
    # after pool layer, the matrix size = (28/2)*(28/2) = 14*14 for every channel
    # when flatten the feature to linear, we get 128*14*14 = 25088
    # fc1（全連接層1）, input size: 1*25088, output size: 1*256
    self.PictureSize -= 2*2 # two conv layer with kernel=3 stride = 1
    self.PictureSize = int(self.PictureSize/2) # pool layer 2*2 
    self.fc1 = nn.Linear(128*self.PictureSize*self.PictureSize , 256)
    # fc2（全連接層2）, input size: 1*256, output size: 1*10
    self.fc2 = nn.Linear(256, 10)

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
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(25088, 256)
        self.fc2 = nn.Linear(256, 10)

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
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
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