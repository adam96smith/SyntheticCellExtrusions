#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: u5501629
"""
#%% packages 
import matplotlib.pyplot as plt, numpy as np
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm
from medmnist import INFO
from medmnist.dataset import OrganMNIST3D
from numpy import random
import hiddenlayer as hl
import albumentations as A
import argparse

# pytorch
import torch 
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary
from torchvision.transforms import v2

# tensor board support for monitoring training and validation
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def seed_everything(seed):
    '''
    Apply `seed` to numpy, random and torch.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
#%% parser params
parser= argparse.ArgumentParser(description='Train a 3D Extrusion Detection Model')
parser.add_argument(
    "--data-dir", type=Path, required=True,
    help="Folder containing the 'extrusion' and 'control' data folders."
)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--data-set-size", type=int, required=True, help="Maximum number of TIFF volumes loaded per class.")
args = parser.parse_args()
#%% params
validation_ratio = 0.3
epochs = args.epochs
batch_size = 4
vbatch_size = 4
p = 0.25 # probablility of transforms being applied
crop_val = 10
data_set_size = args.data_set_size
z_depth_val = 0

seed_everything(42)
#%% custom augmentations 

# class GaussianNoise:
    
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
        
#     def __call__(self, x): # find out what to do with the mean and std from adam 
#         noise = torch.randn_like(x) * self.std + self.mean
#         return (x + noise)

class GaussianBlur3D:
    
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma

        # kernal
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
        kernel = kernel / kernel.sum()
        self.kernel = kernel.view(1,1,kernel_size,kernel_size,kernel_size)

    def __call__(self, x):
        
        padding = self.kernel_size // 2
        # print(f"Shape: {x.shape}")
        # print(f"Type: {x.dtype}")
        x = nn.functional.conv3d(x, self.kernel.to(x.dtype), padding=padding)
        return x

class Int_rescale:
    
    def __init__(self, max_scale, max_bias):
        self.max_bias = max_bias
        self.max_scale = max_scale
    
    def __call__(self, img_tensor):
        img_mean = 0
        sigma = 1
        
        img_norm = (img_tensor - img_mean)/sigma
        
        scale_value = random.uniform(1- self.max_scale, 1+ self.max_scale)
        bias_value = random.uniform(high= self.max_bias)
        
        img_scaled = (img_norm * scale_value) + bias_value
        img_aug = img_scaled * sigma + img_mean
        
        return img_aug
    

#%% Model architecture

class model(nn.Module):

    def __init__(self):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv3d(
            1, 16,
            kernel_size=(1,3,3),
            padding=(0,1,1)
        )

        self.conv2 = nn.Conv3d(
            16, 32,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv3d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.conv4 = nn.Conv3d(
            64, 128,
            kernel_size=3,
            padding=1
        )

        # batch norms
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(128)

        # classifer
        self.fc = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=(1,2,2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=(2,2,2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, kernel_size=(2,2,2))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool3d(x, 1)
        x = x.flatten(1)
        x = self.fc(x)

        return x
    
    
zednet = model()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
zednet = zednet.to(device)

#%% data set 

# data storage
class MYDataset(Dataset):
    
    def __init__(self, X, Y, transforms):
        self.X = X
        self.Y = Y
        self.transforms = transforms
    
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, index):

        image = self.X[index]   # numpy array (16,50,50)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        image = torch.tensor(image).unsqueeze(0)
        label = torch.tensor(self.Y[index]).unsqueeze(0)

        return image, label
 
 
my_transforms = A.Compose([
    A.HorizontalFlip(p=p),
    A.VerticalFlip(p=p),
    A.RandomBrightnessContrast(p=p),
    # A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.8, 0.8), p=p),
    A.Normalize(mean=0, std=1, max_pixel_value=255.0)
])

# class_a = 0
# class_b = 1
# # size = 100
# # val_split = 0.3

# # valsize = size*val_split
# # trainsize = size-valsize

# all_train = OrganMNIST3D(split="train", download=True, size= 64)
# all_val   = OrganMNIST3D(split="val", download=True, size= 64)

# def filter_classes(dataset):
#     imgs = []
#     labels = []
    
#     for img, label in dataset:
#         label = int(label)
        
#         if label == class_a:
#             imgs.append(img)
#             labels.append(0)
#         elif label == class_b:
#             imgs.append(img)
#             labels.append(1)
    
#     imgs = torch.Tensor(imgs)
#     labels = torch.tensor(labels).unsqueeze(1).float()
    
#     return imgs, labels

# X_train, Y_train = filter_classes(all_train)
# X_val,   Y_val   = filter_classes(all_val)

# X_train = X_train[:, :, ::4, :, :]
# X_val = X_val[:, :, ::4, :, :]

# Synth_dataset_train = MYDataset(X_train, Y_train, transforms=my_transforms)
# Synth_dataset_val   = MYDataset(X_val,   Y_val,   transforms=my_transforms)

# training_loader = DataLoader(Synth_dataset_train, batch_size=batch_size, shuffle=True)
# validation_loader = DataLoader(Synth_dataset_val, batch_size=vbatch_size, shuffle=False)


# read in data get labels and store in Synthetic_Dataset class
data_dir = Path(args.data_dir).expanduser()
# print(f"{data_dir}")
Syth_E_dir = data_dir / 'extrusion'
Syth_C_dir = data_dir / 'control'

paths_e = [f for f in Syth_E_dir.glob("*.tif") if not f.name.startswith(".")]
paths_c = [f for f in Syth_C_dir.glob("*.tif") if not f.name.startswith(".")]
num_e = len(paths_e)
num_c = len(paths_c)

X = []
Y = []

for i in (paths_e, paths_c):
    count = 0 #
    for f in i:
        
        if count >= data_set_size:
            break
        
        # normalisation
        pre_norm = tiff.imread(f) 
        # post_norm = ((pre_norm/255.0) - 0.5)-0.5                                   # (pre_norm - np.min(pre_norm)) / (np.max(pre_norm) - np.min(pre_norm))
        # post_norm = pre_norm/255.0
        cropped = pre_norm[z_depth_val:-z_depth_val if z_depth_val > 0 else None, crop_val:-crop_val, crop_val:-crop_val]
        
        # if i == paths_e: # debug
        #     cropped = cropped * -1
            
        X.append(cropped)
        
        if i == paths_e:
            Y.append(1)
        else:
            Y.append(0)
            
        count += 1


Synth_dataset = MYDataset(X, Y, transforms= my_transforms)

# making training and validation sets
dataset_size = len(Synth_dataset)
val_size = int(dataset_size * validation_ratio)
train_size = dataset_size - val_size

# random seed to reproduce training validation split. 
seed = torch.Generator().manual_seed(42)
training_data, validation_data = random_split(Synth_dataset, [train_size, val_size], generator= seed)

# print(f"Training samples: {len(training_data)}")
# print(f"Validation samples: {len(validation_data)}")

# data loaders 
training_loader = DataLoader(training_data, batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, vbatch_size, shuffle=False)


#%% training

# use stochastic gradient decent for optimisation
optimiser = torch.optim.Adam(zednet.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimiser,
    base_lr=1e-6,
    max_lr=1e-3,
    step_size_up=500,
    step_size_down=1500,
)

loss_function = nn.BCEWithLogitsLoss(reduction='mean')

def train_one_epoch(epoch_idx, tb_writer):
    running_loss = 0
    # last_loss = 0
    correct = 0
    total = 0
    
    for i, data in enumerate(training_loader):
        
        inputs, labels = data
        
        # move training data to GPU 
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)
        
        # zero the gradient for each batch to prevent buildup
        optimiser.zero_grad()
        # find outputs predicited by our model
        outputs = zednet(inputs)
        
        # if i == 0: # log feature maps and filters to TB
        #     log_feature_maps(writer, epoch_number)
        
        # if epoch_idx == 0 and i == 0:

        #     input_vol = activations["input"][0]
        #     input_vol = input_vol.cpu().numpy()
        #     input_vol = input_vol[0]

            # tiff.imwrite("debug_input_stack.tif", input_vol.astype("float32"))
            
        # calculate loss and gradients
        loss = loss_function(outputs, labels.float())
        loss.backward()
        # adjust weights accordingly
        optimiser.step()
        scheduler.step()  
        # update running loss
        running_loss += loss.item()
        
        preds = torch.sigmoid(outputs)  # because BCEWithLogitsLoss
        preds = (preds > 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.numel()
        
    # log epoch loss and acc to tensor board    
    epoch_loss = running_loss / len(training_loader)
    epoch_acc = correct / total
    tb_writer.add_scalar("Loss/train", epoch_loss, epoch_idx)
    tb_writer.add_scalar("Acc/train", epoch_acc, epoch_idx)
        
    return epoch_loss, epoch_acc
            

#####################################################training loop#####################################################################################

#initilise tensor board and epoch number
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/extrusion_trainer_{}'.format(timestamp))
epoch_number = 0
best_vloss = 1_000_000.

# training loop 
for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))
    
    zednet.train(True)
    avg_loss, avg_acc = train_one_epoch(epoch_number, writer)
    zednet.train(False)
    
    # run validation + init params
    running_vloss = 0.0
    correct = 0
    total = 0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        
        vinputs = vinputs.float().to(device)
        vlabels = vlabels.float().to(device)
        
        voutputs = zednet(vinputs)
        vloss = loss_function(voutputs, vlabels)
        running_vloss += vloss.item()
        
        # accuracy 
        vpreds = torch.sigmoid(voutputs)
        vpreds = (vpreds > 0.5).float()
        
        correct += (vpreds == vlabels).sum().item()
        total += vlabels.numel()
        
    avg_vloss = running_vloss / len(validation_loader)
    val_acc = correct / total
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.add_scalars(
    'Accuracy',
    {'train': avg_acc, 'val': val_acc},
    epoch_number + 1)
    writer.flush()
    
    # track best performance
    if avg_vloss < best_vloss:
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        # delete old models for this run
        for f in model_dir.glob(f"model_{timestamp}_*.pt"):
            f.unlink()
        
        # save the new best model
        model_path = model_dir / f"model_{timestamp}_{epoch_number+1}.pt"
        torch.save(zednet.state_dict(), model_path)
        best_vloss = avg_vloss  

    epoch_number += 1

