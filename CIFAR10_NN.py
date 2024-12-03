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
from tqdm import tqdm

class CIFARDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, idx):
        return(self.data[idx])
    
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dir = '/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/CIFAR10_NeuralNetworkd/Dataset/Train'
test_dir = '/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/CIFAR10_NeuralNetworkd/Dataset/Test'

train_dataset = CIFARDataset(train_dir, transform)
valid_size = len(train_dataset) // 5
train_size = len(train_dataset) - valid_size

test_dataset = CIFARDataset(test_dir,transform)
valid_dataset, train_dataset = random_split(train_dataset, [valid_size,train_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input: 3x32x32 -> Output: 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Input: 32x32x32 -> Output: 64x32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Input: 64x32x32 -> Output: 128x32x32
        
        # Max pooling layer (applies after each convolution)
        self.pool = nn.MaxPool2d(2, 2)  # Halves the spatial dimensions (e.g., from 32x32 to 16x16)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Flattened dimensions after 3 pooling layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # Output 10 classes for CIFAR-10

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutions, ReLU activations, and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pool

        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 4 * 4)  # Flattening the tensor (adjusted to match the correct size)

        # Apply fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout to the activations
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation function here as it's for classification)

        return x


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleClassifier()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
loss_track = []


for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc = 'Training'):
        # Move data to device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Optimize the parameters
        optimizer.step()

        # Accumulate loss for logging
        running_loss += loss.item()
    loss_track.append(running_loss/len(train_loader))


    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

print(loss_track)
plt.plot([n for n in range(epochs)], loss_track)
plt.show()