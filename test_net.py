import torch
from Models.net import RLModel

model = RLModel([(1, 42, 8, 8), (1, 10, 8, 8)], [(73*64,), 1])
input_tensor = torch.randn(1, 1, 42, 64)
policy, value = model(input_tensor)
print(policy.shape, value.shape)
