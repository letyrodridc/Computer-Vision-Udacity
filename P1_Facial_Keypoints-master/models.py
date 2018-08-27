## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        fc1_input_size = 43264 #34*52*52 #
        self.fc1 = nn.Linear(fc1_input_size, 10000)
        self.fc2 = nn.Linear(10000, 1500)
        self.fc3 = nn.Linear(1500, 136)
        
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.dropout(self.pool2(F.relu(self.conv2(x))), p=0.1)
        x = F.dropout(self.pool3(F.relu(self.conv3(x))), p=0.2)
        x = F.dropout(self.pool4(F.relu(self.conv4(x))), p=0.3)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        x = F.dropout(x, p=0.2)
        x = self.fc3(x)
        
        return x
