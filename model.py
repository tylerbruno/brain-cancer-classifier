import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

class BrainTumorMRIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    

dataset = BrainTumorMRIDataset(
    data_dir='/kaggle/input/brain-tumor-mri-dataset/Training'
)

class BrainTumorClassifer(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

# Get a dictionary associating target values with folder names
data_dir = '/kaggle/input/brain-tumor-mri-dataset/Training'
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_dir = '/kaggle/input/brain-tumor-mri-dataset/Training'
dataset = BrainTumorMRIDataset(data_dir, transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = BrainTumorClassifer(num_classes=4)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_folder = '/kaggle/input/brain-tumor-mri-dataset/Training'
test_folder = '/kaggle/input/brain-tumor-mri-dataset/Testing'

train_dataset = BrainTumorMRIDataset(train_folder, transform=transform)
test_dataset = BrainTumorMRIDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Simple training loop
num_epochs = 5
train_losses, test_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BrainTumorClassifer(num_classes=4)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        # Move inputs and labels to the device
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
         
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    test_loss = running_loss / len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {test_loss}")
    
# Plot the loss over epochs
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Testing loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()