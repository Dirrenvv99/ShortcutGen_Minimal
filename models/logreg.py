import os
import torch
import numpy as np
from tqdm import tqdm
import util
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
         x = torch.flatten(x,start_dim=1)
         outputs = torch.sigmoid(self.linear(x))
         return outputs

