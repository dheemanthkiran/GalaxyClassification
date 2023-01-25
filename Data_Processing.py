import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    """Custom Dataset class with Galaxy images"""

    def __init__(self,mapping_file, img_dir, img_infoFile, transform=transforms.ToTensor(), target_transform=None,  device="cuda:0"):
        """Instanciante dataset subclass

            Parameters:
                mapping file (string) directory of mapping file
                img_dir (string) directory of file with images
                img_infoFile (String) directory of
            Returns:
                None
        """
        self.img_labels = pd.read_csv(mapping_file)
        self.img_dir = img_dir
        self.img_info = pd.read_csv(img_infoFile)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns: len(int) Lenght of dataset"""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        returns Image and Image label of the indexed item in a combined tensor

        Parameters:
                idx(int) index of item
        Returns:
            Tensor: Normalised pixel values of the image
            np.array: Classification Lablel of Image
        """
        """Locating Image path"""
        i = 0
        list = [1,2,3]
        while True:
            try:
                """Tries to fetch Image with requested id"""
                img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx + i, 2]) + '.jpg')
                image = Image.open(img_path)
                image_id = self.img_labels.iloc[idx + i, 0]
                """Loading fraction of votes for each category"""
                list = [self.img_info[self.img_info['dr7objid'] == image_id].iloc[0, 12],
                        self.img_info[self.img_info['dr7objid'] == image_id].iloc[0, 18],
                        self.img_info[self.img_info['dr7objid'] == image_id].iloc[0, 24]]
                break
            except FileNotFoundError:
                """Some Images are missing. If the file is not found, it will just return the next available image"""
                if i == self.__len__():
                    i = 0
                else:
                    i = i + 1
                pass
            except IndexError:
                """A small number of images have unindexed information. This handles that error"""
                if i == self.__len__():
                    i = 0
                else:
                    i = i + 1
                pass

        """creating new label with the highest voted category"""
        label = torch.zeros(3)
        label[np.argmax(list)] = 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = image[:, 84:339, 84:339]
        image.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        return image, label
        """The except is there incase the id requested does not have an accosiated image, as some images are missing"""




"""#Creating Galaxy Dataset
Galaxy_dataset = CustomImageDataset(
    mapping_file="./gz2_filename_mapping.csv",
    img_dir="./images_gz2/images",
    img_infoFile="./gz2_hart16.csv")

image, data = Galaxy_dataset.__getitem__(35)


print(image.size())


# Creating new instance of dataloader class

dataloader = DataLoader(dataset=Galaxy_dataset, batch_size=10, shuffle=True)
# Creating iterator for dataloader
dataiter = iter(dataloader)
data = dataiter.__next__()
features, labels = data
print(features, labels)"""
