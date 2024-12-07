import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import PIL
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class CIFARDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, idx):
        return(self.data[idx])
    
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(2 * 2 * 128, 256)  # Increased fully connected size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool(self.relu4(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleClassifier()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
max_epochs = 50
patience = 5
epoch = 0
min_delta = 1e-1
best_loss = float('inf')
loss_track = []
valid_track= []

while epoch < max_epochs:
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

        # Update the scheduler
    scheduler.step()
    loss_track.append(running_loss/len(train_loader))

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc = 'Validation'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            valid_loss += loss.item()
        valid_track.append(valid_loss/len(valid_loader))
        print(f"Epoch {epoch+1} | Train Loss: {running_loss/len(train_loader):.4f} | Valid Loss: {valid_loss/len(valid_loader)}")

    # Check for improvement in validation loss
    if valid_loss < best_loss - min_delta:
        # Significant improvement
        best_loss = valid_loss
        patience_counter = 0  # Reset patience
    else:
        # No significant improvement
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
    print(f'Patience: {patience_counter}')
    epoch += 1


plt.plot([n for n in range(len(loss_track))], loss_track, label="Training Loss")
plt.plot([n for n in range(len(valid_track))], valid_track, label="Validation Loss")
plt.xlabel("Epcochs")
plt.ylabel("Cross Entropy Loss")
plt.grid()
plt.legend()
plt.show()

# Set the model to evaluation mode
model.eval()

# Sample random indices from the test dataset
random_indices = random.sample(range(len(test_dataset)), 9)

# Extract the random images and labels
random_images = []
random_labels = []


for idx in random_indices:
    image, label = test_dataset[idx]  # Access the image and label from the dataset
    random_images.append(image)
    random_labels.append(label)

# Move images to the correct device for the model
random_images = torch.stack(random_images).to(device)

# Get model predictions
with torch.no_grad():
    outputs = model(random_images)
    probabilities = F.softmax(outputs, dim=1)
    _, predicted_classes = torch.max(probabilities, 1)

# Plot the randomly selected images with predictions
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for i in range(9):
    ax = axes[i // 3, i % 3]
    image = random_images[i].cpu().numpy().transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.axis('off')

    # Get the predicted class and its probability
    predicted_class = predicted_classes[i].item()
    probability = probabilities[i, predicted_class].item()

    # True label for the image
    true_label = random_labels[i]

    # Show predicted class, true label, and probability
    ax.set_title(f'Pred: {predicted_class} ({probability*100:.2f}%)\nTrue: {true_label}')

plt.tight_layout()
plt.show()
