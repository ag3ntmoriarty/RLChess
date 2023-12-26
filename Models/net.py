import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.policy_net import PolicyNet
from Models.value_net import ValueNet

class RLModel(nn.Module):
    """
    RLModel for predicting the value of a board

    INPUTS:

        input_size: [64x18x1, 64x10x1]
    
        output_size: [64x72x1, 1]
        
    """
    def __init__(self, input_size, output_size):
        super(RLModel,self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.policy_net = PolicyNet(input_size[0], output_size[0])
        self.value_net = ValueNet(input_size[1], output_size[1])
        
    def forward(self, input):
        policy = self.policy_net(input)
        value = self.value_net(input[:10])
        return policy, value