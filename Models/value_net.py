import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    """ValueNet for predicting the value of a board
    
    INPUTS:
    
        input_size: 64x10x1
        
        output_size: 1
        
    """
    def __init__(self, input_size, output_size):
        super(ValueNet,self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)       
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(73*8*8, 73*8*8)
        self.linear2 = nn.Linear(73*8*8, self.output_size[1])
    
    def forward(self, input):
        x = self.bn1(self.conv1(input))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.tanh(self.linear2(x)) #tanh for value in range [-1,1]
        return x