import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

w = torch.tensor(2.0, requires_grad=True, device='cuda')

y = w**2
z = 2*y+5

z.backward()
print('기울기 미분값 {}'.format(w.item()))






