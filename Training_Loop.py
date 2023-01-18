import Network
import Data_Processing
import torch.utils.data
import torch

Galaxy_dataset = Data_Processing.CustomImageDataset(
    mapping_file="./gz2_filename_mapping.csv",
    img_dir="./images_gz2/images",
    img_infoFile="./gz2_hart16.csv")

Training_Data, Validation_Data = torch.utils.data.random_split(dataset=Galaxy_dataset, lengths=[0.85, 0.15], generator=torch.Generator().manual_seed(12))

print(Training_Data.__len__())
print(Validation_Data.__len__())