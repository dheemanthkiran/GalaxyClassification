import torch.nn.functional as F
import torch
import Data_Processing


class LeNet(torch.nn.Module):

    def __init__(self):
        """
        Initiates layers of the Network. Architecture used is LeNet5

        Parameters:
            self: LeNet 5 object
        Returns:
            No return
        """
        super(LeNet, self).__init__()
        # 2 convolusional layers to categorize features of Image
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        #3 Linear layers for final output
        self.fc1 = torch.nn.Linear(16 * 25 * 25, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 3)

    def forward(self, x):
        """
        does a forwads pass of the network

        Parameters: ([3,n,n] tensor object) contains the image rgb values
        """
        # Max pooling over a (4,4) window to decrease parameters
        x = F.max_pool2d(F.relu(self.conv1(x)), 4)
        x = F.max_pool2d(F.relu(self.conv2(x)), 4)
        #Flattens the 3d tensor into a 1d vector for Lin Layers
        x = torch.flatten(x, 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


"""model = LeNet()


Galaxy_dataset = Data_Processing.CustomImageDataset(
    mapping_file="./gz2_filename_mapping.csv",
    img_dir="./images_gz2/images",
    img_infoFile="./gz2_hart16.csv")

Image, Label = Galaxy_dataset.__getitem__(35)

a = model.forward(Image)

print(a)"""

