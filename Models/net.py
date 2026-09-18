import torch
import torch.nn as nn
from Models.policy_net import PolicyHead
from Models.value_net import ValueHead

class RLModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=512, nhead=8, num_layers=10):
        super(RLModel, self).__init__()
        # input_size is like [(1, 42, 8, 8), (1, 10, 8, 8)]
        # We use the full 42 channel input for the shared backbone
        self.in_channels = input_size[0][1] # 42
        
        # Project 42 channels into d_model for each of the 64 squares
        self.input_proj = nn.Conv2d(self.in_channels, d_model, kernel_size=1)
        
        # Positional Encoding for the 64 squares
        self.pos_encoder = nn.Parameter(torch.randn(1, 64, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            norm_first=True # Better training stability for modern transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Flattened size after transformer
        flattened_size = d_model * 64
        
        # Heads
        self.policy_head = PolicyHead(in_features=flattened_size, output_size=output_size[0][0])
        self.value_head = ValueHead(in_features=flattened_size, output_size=output_size[1])
        
    def forward(self, x):
        # x is (B, 42, 8, 8)
        B = x.size(0)
        
        # 1. Project channels
        x = self.input_proj(x) # (B, d_model, 8, 8)
        
        # 2. Flatten spatial dimensions -> (B, d_model, 64) -> transpose -> (B, 64, d_model)
        x = x.view(B, -1, 64).transpose(1, 2)
        
        # 3. Add positional encoding
        x = x + self.pos_encoder
        
        # 4. Transformer
        x = self.transformer(x) # (B, 64, d_model)
        
        # 5. Flatten for heads
        x = x.reshape(B, -1) # (B, 64 * d_model)
        
        # 6. Heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value