import torch
from torchinfo import summary
import Network
import Data_Processing
from torch.utils.data import DataLoader
import numpy as np

model = Network.LeNet()
checkpoint = torch.load("my_model_Adam.pth.tar")
model.load_state_dict(checkpoint['state_dict'])


def train():
    Galaxy_dataset = Data_Processing.CustomImageDataset(
        mapping_file="./gz2_filename_mapping.csv",
        img_dir="./images_gz2/images",
        img_infoFile="./gz2_hart16.csv")

    Training_Data, Validation_Data, Test_Data = torch.utils.data.random_split(dataset=Galaxy_dataset,
                                                                              lengths=[0.6, 0.2, 0.2],
                                                                              generator=torch.Generator().manual_seed(
                                                                                  12))
    loader = DataLoader(Test_Data, batch_size=256, shuffle=True, num_workers=2)
    model.train(False)
    torch.set_grad_enabled(False)
    correct = 0
    wrong = 0
    for i, data in enumerate(loader):
        input, label = data[0], data[1]
        output = model(input)
        print(i)
        if i % 20 == 19:
            print("Right: ", correct, "  Wrong:", wrong)
        for j in range(128):
            if np.argmax(output[j]) == np.argmax(label[j]):
                correct = correct + 1
            else:
                wrong = wrong + 1

    accuracy = correct / (correct + wrong)
    print(accuracy)

if __name__ == '__main__':
    train()