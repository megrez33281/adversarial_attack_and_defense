import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets
import Get_test


class Net(nn.Module):
  # neural network model
  def __init__(self):
    super(Net, self).__init__()
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
    self.fc1 = nn.Linear(9216, 128)
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

def fit(model,device,train_loader,val_loader,epochs, criterion, optimizer, scheduler):
  data_loader = {'train':train_loader,'val':val_loader}
  print("Fitting the model...")
  train_loss,val_loss=[],[]
  for epoch in range(epochs):
    loss_per_epoch,val_loss_per_epoch=0,0
    for phase in ('train','val'):
      for i,data in enumerate(data_loader[phase]):
        input,label  = data[0].to(device),data[1].to(device)
        output = model(input)
        #calculating loss on the output
        loss = criterion(output,label)
        if phase == 'train':
          optimizer.zero_grad()
          #grad calc w.r.t Loss func
          loss.backward()
          #update weights
          optimizer.step()
          loss_per_epoch+=loss.item()
        else:
          val_loss_per_epoch+=loss.item()
    scheduler.step(val_loss_per_epoch/len(val_loader))
    print("Epoch: {} Loss: {} Val_Loss: {}".format(epoch+1,loss_per_epoch/len(train_loader),val_loss_per_epoch/len(val_loader)))
    train_loss.append(loss_per_epoch/len(train_loader))
    val_loss.append(val_loss_per_epoch/len(val_loader))
  return train_loss,val_loss


def train_model():
    np.random.seed(42)
    torch.manual_seed(42)
    # chang picture value from [0, 255] to [0, 1] and Normalize the picture（mean = 0, standard deviation = 1）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    # download dataset
    dataset = Get_test.get_train_set(transform)

    val_set_size = int(len(dataset)/6)
    train_set_size = len(dataset) - val_set_size
    # split dataset into training set and validation set
    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_size, val_set_size])


    # create loader
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=1,shuffle=True)
    
    use_cuda=True   # use GPU first
    # choose use device
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


    # set model
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.0001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    # train model
    epochs = 10
    loss,val_loss=fit(model,device, train_loader, val_loader, epochs, criterion, optimizer, scheduler)
    fig = plt.figure(figsize=(5,5))
    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(1,epochs+1), loss, "*-",label="Loss")
    plt.plot(np.arange(1,epochs+1), val_loss,"o-",label="Val Loss")
    plt.xlabel("Num of epochs")
    plt.legend()
    plt.show()

    # save model's weight
    torch.save(model.state_dict(), "original_model_weights.pth")


  
def read_model(device):
    model = Net().to(device)  # define model architecture
    model.load_state_dict(torch.load("original_model_weights.pth", weights_only=True))
    model.eval()  # change model mode to evaluation mode (評估模式)
    return model

if __name__ == '__main__':
    train_model()