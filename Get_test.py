import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets

def get_test_loader()->torch.utils.data.DataLoader:
    # 下載測試集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    test_set = datasets.MNIST(root = './data', train=False, transform = transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=True)
    return test_loader

def get_device():
    use_cuda=True   # 優先使用GPU
    # 選擇使用的設備
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    return device