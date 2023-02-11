import Network
import Data_Processing
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import gc
'''Initialising galaxy dataset'''
Galaxy_dataset = Data_Processing.CustomImageDataset(
    mapping_file="./gz2_filename_mapping.csv",
    img_dir="./images_gz2/images",
    img_infoFile="./gz2_hart16.csv")

'''Checking for cuda device availability'''
dev = None
if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

'''Creating model, loss function and moving them to device determined above'''
model = Network.LeNet()
model.to(device=dev)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device=dev)
'''Loading last saved model and optimizer parameters to the initialised models and optimizer'''
checkpoint = torch.load("my_model_Adam.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])

def train():
    epochsNum = 2
    best_vloss = 1000000
    '''Creating data split. Custom seed is used to the random shuffle is reproducable'''
    Training_Data, Validation_Data, Test_Data = torch.utils.data.random_split(dataset=Galaxy_dataset,
                                                                              lengths=[0.6, 0.2, 0.2],
                                                                              generator=torch.Generator().manual_seed(
                                                                                  12))
    validationLoader = DataLoader(Validation_Data, shuffle=False, batch_size=128, pin_memory=True, num_workers=2)
    trainingLoader = DataLoader(Training_Data, shuffle=True, batch_size=128, pin_memory=True, num_workers=2)
    '''training an epoch'''
    for epoch in range(epochsNum):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        torch.set_grad_enabled(True)
        last_loss = 0.
        running_loss = 0

        for i, data in enumerate(trainingLoader):
            print("batch ", i)
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(device=dev)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs).to(device=torch.device("cpu"))
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                last_loss = running_loss / 20  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                # tb_x = epoch_index * len(trainingLoader) + i + 1
                # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        # We don't need gradients on to do reporting
        model.train(False)
        gc.collect()
        torch.cuda.empty_cache()
        torch.set_grad_enabled(False)
        running_vloss = 0.0
        '''calculation validation loss'''
        for i, vdata in enumerate(validationLoader):
            vinputs, vlabels = vdata[0].to(device=dev), vdata[1].to(device=dev)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            print("batch: ", i, "  Loss: vloss")
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(last_loss, avg_vloss))
        # Track best performance, and save the model's state

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}'.format(epochsNum)
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            print("=> Saving model")
            torch.save(checkpoint, "my_model_Adam.pth.tar")
            '''saves model if it performed better than the previous best model'''

        epochsNum += 1

'''This if statement is needed to make multiple workers function on windows'''
if __name__ == '__main__':
    train()


