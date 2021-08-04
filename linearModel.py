import torch
import torch.nn as nn

torch.manual_seed(1)

class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)