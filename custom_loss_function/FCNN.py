import torch
import torch.nn as nn
from torch.autograd.grad_mode import F


class FCNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(FCNN, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = F.relu(self.lin1(x))
        out = self.lin2(x1)
        return out