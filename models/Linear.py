import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class linearModel(nn.Module):
    def __init__(self, in_components = 30, out_components = 10):
        super(linearModel, self).__init__()
        self.layer = nn.Linear(in_components, out_components)
        self.flat = nn.Flatten()
    
    def forward(self, x):
        x = self.flat(x)
        return self.layer(x)

