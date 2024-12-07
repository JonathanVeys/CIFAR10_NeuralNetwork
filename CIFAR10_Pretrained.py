import torch 
from torch.utils.data import dataloader, Dataset, random_split
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

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
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0,5, 0,5]),
    transforms.ToTensor()
])

train_dataset = CIFARDataset(train_dir, transform)
test_dataset = CIFARDataset(test_dir, transform)