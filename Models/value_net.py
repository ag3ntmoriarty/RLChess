import torch
import torch.nn as nn

class ValueHead(nn.Module):
    def __init__(self, in_features, output_size):
        super(ValueHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x