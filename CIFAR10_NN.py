import PIL.ImageShow
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import PIL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CIFARDataset(Dataset):
    def __init__(self, data_dir):
        self.data = ImageFolder(data_dir)

    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, idx):
        return(self.data[idx])
    
train_dir = '/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/CIFAR10_NeuralNetworkd/Dataset/Train'
test_dir = '/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/CIFAR10_NeuralNetworkd/Dataset/Test'

train_dataset = CIFARDataset(train_dir)
valid_size = len(train_dataset) // 5
train_size = len(train_dataset) - valid_size

test_dataset = CIFARDataset(test_dir)
valid_dataset, train_dataset = random_split(train_dataset, [valid_size,train_size])
print(f'Train dataset size = {len(train_dataset)}')
print(f'Valid dataset size = {len(valid_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)

        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))
        x = self.pool(nn.ReLU(self.conv3(x)))
        x = x.view(-1,128*8*8)
        x = nn.ReLU(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU(self.fc2(x))
        x = self.fc3(x)
        return(x)
