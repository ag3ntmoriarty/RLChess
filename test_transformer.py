import torch
from Models.net import RLModel
import chess

model = RLModel([(1, 42, 8, 8), (1, 10, 8, 8)], [(73*8*8,), 1])
input_tensor = torch.randn(2, 42, 8, 8)
policy, value = model(input_tensor)
print("Policy shape:", policy.shape)
print("Value shape:", value.shape)
print("Transformer forward pass successful!")
