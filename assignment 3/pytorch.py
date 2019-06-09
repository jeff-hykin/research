import torch
from torch import nn
import torch.nn.functional as F


class ANet(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        number_of_colors_in_picture = 3
        image_dimensions = (32, 32)
        # Create the layers we're going to use
        self.conv1 = nn.Conv2d(in_channels=number_of_colors_in_picture, out_channels=20, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=5)
        self.fully_connected_layer1 = nn.Linear(in_features=30 * 5 * 5, out_features=300)
        self.fully_connected_layer2 = nn.Linear(in_features=300, out_features=75)
        self.fully_connected_layer3 = nn.Linear(in_features=75, out_features=10)

    def forward(self, incoming_data):
        #
        # Connect all the layers together
        #
        data_after_conv1 = F.relu(self.conv1(incoming_data))
        # dimensions = (28 , 28 , 20)
        data_after_pool1 = self.pool(data_after_conv1)
        # dimensions = (14 , 14 , 20)
        data_after_conv2 = F.relu(self.conv2(data_after_conv1))
        # dimensions = (10 , 10 , 30)
        data_after_pool2 = self.pool(data_after_conv2)
        # dimensions = ( 5 ,  5 , 30)
        flattened_data = data_after_conv2.view(-1, 5 * 5 * 30)
        # dimensions = (750)
        layer1_data = F.relu(self.fully_connected_layer1(flattened_data))
        # dimensions = (300)
        layer2_data = F.relu(self.fully_connected_layer2(layer1_data))
        # dimensions = (75)
        output_data = F.sigmoid(self.fully_connected_layer3(layer2_data))
        # dimensions = (10)
        return x
