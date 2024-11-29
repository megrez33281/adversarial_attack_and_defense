import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import Get_test
import model_architecture_FashionMNIST
import model_architecture_CIFAR10

'''
class Net(model_architecture_FashionMNIST.Net):
   # change model when use different dataset
   {}
'''
class Net(model_architecture_CIFAR10.Net):
   # change model when use different dataset
   {}

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