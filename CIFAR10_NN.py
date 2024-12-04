import PIL.ImageShow
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import PIL

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
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.relu3 = nn.ReLU()
        
        self.pool = nn.MaxPool2d(2, 2)  

        self.fc1 = nn.Linear(128 * 4 * 4, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.pool(self.relu1(self.conv1(x)))  
        x = self.pool(self.relu2(self.conv2(x)))  
        x = self.pool(self.relu3(self.conv3(x)))  

        x = x.view(-1, 128 * 4 * 4) 

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleClassifier()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 25
loss_track = []
valid_track= []

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc = 'Training'):
        # Move data to device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Optimize the parameters
        optimizer.step()

        # Accumulate loss for logging
        running_loss += loss.item()
    loss_track.append(running_loss/len(train_loader))

    model.eval()
    eval_loss = 0.0
    for images, labels in tqdm(valid_loader, desc = 'Validation'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        eval_loss += loss.item()
    valid_track.append(eval_loss/len(valid_loader))
    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f} | Valid Loss: {eval_loss/len(valid_loader)}")

plt.plot([n for n in range(epochs)], loss_track)
plt.plot([n for n in range(epochs)], valid_track)
plt.xlabel("Epcochs")
plt.ylabel("Cross Entropy Loss")
plt.grid()
plt.legend()
plt.show()