import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets


class Net(nn.Module):
  # 類神經網路模型
  def __init__(self):
    super(Net, self).__init__()
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
    train_loss.append(loss_per_epoch/len(train_loader))
    val_loss.append(val_loss_per_epoch/len(val_loader))
  return train_loss,val_loss


def train_model():
    np.random.seed(42)
    torch.manual_seed(42)
    # 將圖片轉黑白、標準化（平均值=0, 標準差=1）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    # 下載資料集
    dataset = datasets.MNIST(root = './data', train=True, transform = transform, download=True)
    # 將資料集分為訓練集、驗證集兩部分
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])


    #創建loader
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=1,shuffle=True)
    
    use_cuda=True   # 優先使用GPU
    # 選擇使用的設備
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


    # 設置model
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.0001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 訓練model
    loss,val_loss=fit(model,device,train_loader,val_loader,10, criterion, optimizer, scheduler)

    # 儲存model參數
    torch.save(model.state_dict(), "original_model_weights.pth")


  
def read_model(device):
    model = Net().to(device)  # 重新定義模型架構
    model.load_state_dict(torch.load("original_model_weights.pth", weights_only=True))
    model.eval()  # 將模型切換到評估模式（非訓練模式）
    return model

if __name__ == '__main__':
    train_model()