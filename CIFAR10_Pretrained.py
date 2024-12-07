import torch 
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torch.optim as optim
import tqdm as tqdm


train_dir = '/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/CIFAR10_NeuralNetworkd/Dataset/Train'
test_dir = '/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/CIFAR10_NeuralNetworkd/Dataset/Test'

class CIFARDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return(self.data[idx])
    
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


train_dataset = CIFARDataset(train_dir, transform)
train_dataset_len = len(train_dataset) // 5
valid_dataset_len = len(train_dataset) - train_dataset_len

test_dataset = CIFARDataset(test_dir, transform)
test_dataset, valid_dataset = random_split(train_dataset, [train_dataset_len, valid_dataset_len])

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class SimpleClasifier(nn.Module):
    def __init__(self):
        super(SimpleClasifier, self).__init__()
        model = resnet18(pretrained=True)
        self.fc = nn.Linear(model.fc.in_features, 10)
        model.to(device)
    
    def forward(self, x):
        return(self.model(x))
    
model = SimpleClasifier()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1
train_loss_track = []
valid_loss_track = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc = 'Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss_track.append(train_loss/len(train_loader))

    with torch.no_grad():
        valid_loss = 0.0
        for images, labels in tqdm(valid_loader, desc = 'Validation'):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            valid_loss += loss.item()
        valid_loss_track.append(valid_loss/len(valid_loader))
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Valid Loss: {valid_loss/len(valid_loader)}")