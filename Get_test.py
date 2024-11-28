import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets

def get_test_set(transform):
    # download test dataset
    #test_set = datasets.MNIST(root = './data', train=False, transform = transform, download=True)
    test_set = datasets.FashionMNIST(root = './data', train=False, transform = transform, download=True)
    return test_set

def get_train_set(transform):
    # download train dataset
    #train_set = datasets.MNIST(root = './data', train=True, transform = transform, download=True)
    train_set = datasets.FashionMNIST(root = './data', train=True, transform = transform, download=True)
    return train_set

def get_test_loader()->torch.utils.data.DataLoader:
    # generate test loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    test_set = get_test_set(transform)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=True)
    return test_loader

def get_device():
    use_cuda=True   # use GPU first
    # choose device
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    return device