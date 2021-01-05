from torch.nn.parameter import Parameter
import torch

weight = Parameter(torch.FloatTensor(5, 4))
print(weight)