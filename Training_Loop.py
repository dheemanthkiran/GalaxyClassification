import Network
import Data_Processing
import torch.utils.data
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim

Galaxy_dataset = Data_Processing.CustomImageDataset(
    mapping_file="./gz2_filename_mapping.csv",
    img_dir="./images_gz2/images",
    img_infoFile="./gz2_hart16.csv")

Training_Data, Validation_Data, Test_Data = torch.utils.data.random_split(dataset=Galaxy_dataset,
                                                                          lengths=[0.6, 0.2, 0.2],
                                                                          generator=torch.Generator().manual_seed(12))

trainingLoader = DataLoader(Training_Data,shuffle=True, batch_size=64)
validationLoader = DataLoader(Validation_Data,shuffle=False, batch_size=64)

model = Network.LeNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.3, dampening=0.1)




def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(trainingLoader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(trainingLoader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


epochsNum = 10


for epoch in range(epochsNum):
    print("Epoch {}:".format(epoch))





