import os
import configparser
from PIL import Image

import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
from torchvision import transforms


config = configparser.ConfigParser()

# path to 'config.ini' file in Google
path = 'drive/Shareddrives/CSEN240_Group11/'

if not os.path.isdir(path):
# for local machine
    path = 'configure.ini'
else:
# for Google CoLab
    path += 'configure.ini'

config.read(path)

root_path = config['PATHS']['root']
train_path = root_path + config['PATHS']['train']

# load dataset
data_dir = os.path.join(root_path, 'torchvision_dataset')
dataset = datasets.ImageFolder(data_dir, transform=lambda img: np.array(img).astype(float))

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    images: torch.Tensor
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

batch_size = 32
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
mean, std = get_mean_std(loader)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 56 * 56, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

model = model.double()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    
    inputs: torch.Tensor
    for i, (inputs, labels) in enumerate(loader):
        print(inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")

model_dst = os.path.join(root_path, 'nn_classify.pth')
os.makedirs(model_dst, exist_ok=True)
torch.save(model.state_dict(), model_dst)
print('model saved successfully')