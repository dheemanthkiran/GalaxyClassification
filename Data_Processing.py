import inspect

import numpy
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, mapping_file, img_dir, img_infoFile, transform=transforms.ToTensor(), target_transform=None):
        self.img_labels = pd.read_csv(mapping_file)
        self.img_dir = img_dir
        self.img_info = pd.read_csv(img_infoFile)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        try:
            img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 2]) + '.jpg')
            image = Image.open(img_path)
            image_id = self.img_labels.iloc[idx, 0]
            list  = [self.img_info[self.img_info['dr7objid'] == image_id].iloc[0, 12],
                    self.img_info[self.img_info['dr7objid'] == image_id].iloc[0, 18],
                    self.img_info[self.img_info['dr7objid'] == image_id].iloc[0, 24]]
            label = torch.zeros(3)
            label[np.argmax(list)] = 1
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        except:
            img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 2]) + '.jpg')
            image = Image.open(img_path)
            image_id = self.img_labels.iloc[idx, 0]
            list = [self.img_info[self.img_info['dr7objid'] == image_id].iloc[0, 12],
                    self.img_info[self.img_info['dr7objid'] == image_id].iloc[0, 18],
                    self.img_info[self.img_info['dr7objid'] == image_id].iloc[0, 24]]
            label = torch.zeros(3)
            label[np.argmax(list)] = 1
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label


Galaxy_dataset = CustomImageDataset(
    mapping_file="./gz2_filename_mapping.csv",
    img_dir="./images_gz2/images",
    img_infoFile="./gz2_hart16.csv")

dataloader = DataLoader(dataset=Galaxy_dataset, batch_size=4, shuffle=True)

dataiter = iter(dataloader)
data = dataiter.__next__()
features, labels = data
print(features, labels)
