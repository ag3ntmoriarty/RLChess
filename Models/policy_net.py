import torch
import torch.nn as nn

class PolicyHead(nn.Module):
    def __init__(self, in_features, output_size):
        super(PolicyHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 4096)
        self.fc2 = nn.Linear(4096, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        # Return raw logits — softmax is applied externally:
        #   - During training: F.cross_entropy() applies log_softmax internally
        #   - During inference (MCTS): we apply softmax explicitly where needed
        return self.fc2(x)
