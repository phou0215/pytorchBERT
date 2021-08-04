import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.LongTensor([[1], [2], [3]]).to(torch.device('cuda'))
y_train = torch.LongTensor([[2], [5], [7]]).to(torch.device('cuda'))

epoch = 1000
w = torch.randn(1, requires_grad=True, device='cuda')
b = torch.randn(1, requires_grad=True, device='cuda')
optimizer = optim.SGD([w, b], lr=0.01)

for e in range(epoch):
    hypothesis = (x_train * w + b)
    loss = torch.mean((hypothesis - y_train) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 100 == 0:
        print('Epoch : {}/{} w: {} b: {} Cost:{}'.format(e, epoch, round(w.item(),2), round(b.item(),2), round(loss.item(),2)))





