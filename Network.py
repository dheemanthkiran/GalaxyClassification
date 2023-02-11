import torch.nn.functional as F
import torch
import Data_Processing
from torch.utils.data import DataLoader


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
        '''Two convolusion layers'''
        # 3 Linear layers for final output
        self.fc1 = torch.nn.Linear(16 * 15 * 15, 120)
        self.fc2 = torch.nn.Linear(120, 60)
        self.fc3 = torch.nn.Linear(60, 3)
        '''Three fully connected layers'''

    def forward(self, x):
        """
        does a forwads pass of the network

        Parameters: ([3,n,n] tensor object) contains the image rgb values
        """
        batch = False
        if len(x.size()) == 4:
            '''Checks if the input os a single image of batch of images'''
            batch = True
        # Max pooling over a (4,4) window to decrease parameters
        x = F.max_pool2d(F.relu(self.conv1(x)), 4)
        x = F.max_pool2d(F.relu(self.conv2(x)), 4)
        # Flattens the 3d tensor into a 1d vector for Lin Layers
        if batch == True:
            x = torch.flatten(x, 1)
        else:
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



