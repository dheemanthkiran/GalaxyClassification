import Network
import Data_Processing
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
from datetime import datetime

Galaxy_dataset = Data_Processing.CustomImageDataset(
    mapping_file="./gz2_filename_mapping.csv",
    img_dir="./images_gz2/images",
    img_infoFile="./gz2_hart16.csv")

Training_Data, Validation_Data, Test_Data = torch.utils.data.random_split(dataset=Galaxy_dataset,
                                                                          lengths=[0.6, 0.2, 0.2],
                                                                          generator=torch.Generator().manual_seed(12))

trainingLoader = DataLoader(Training_Data, shuffle=True, batch_size=128, pin_memory=True)
validationLoader = DataLoader(Validation_Data, shuffle=False, batch_size=16, pin_memory=True)

model = Network.LeNet()
model.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.3, dampening=0.1)

"""for inputs, labels in trainingLoader:
    inputs, labels = inputs.to(device), labels.to(device)

print("Done loading trainer")
for inputs, labels in validationLoader:
    inputs, labels = inputs.to(device), labels.to(device)
print("Done loading validation")"""


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(trainingLoader):
        print("batch ", i)
        # Every data instance is an input + label pair
        inputs, labels = data[0].to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')), data[1].to(
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

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
        if i % 20 == 19:
            last_loss = running_loss / 20  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(trainingLoader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


epochsNum = 1
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

best_vloss = 1_000_000

for epoch in range(epochsNum):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epochsNum)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validationLoader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    """
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epochsNum + 1)
    writer.flush()"""

    # Track best performance, and save the model's state

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}'.format(epochsNum)
        checkpoint = {'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
        print("=> Saving model")
        torch.save(checkpoint, "my_model.pth.tar")

    epochsNum += 1
