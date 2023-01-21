import Network
import Data_Processing
import torch.utils.data
from torch.utils.data import DataLoader
import torch

Galaxy_dataset = Data_Processing.CustomImageDataset(
    mapping_file="./gz2_filename_mapping.csv",
    img_dir="./images_gz2/images",
    img_infoFile="./gz2_hart16.csv")

Training_Data, Validation_Data, Test_Data = torch.utils.data.random_split(dataset=Galaxy_dataset,
                                                                          lengths=[0.6, 0.2, 0.2],
                                                                          generator=torch.Generator().manual_seed(12))

trainingLoader = DataLoader(Training_Data,shuffle=True, batch_size=64)
trainingLoader = DataLoader(Validation_Data,shuffle=False, batch_size=64)
