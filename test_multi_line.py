import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(3)

x_train = torch.FloatTensor([[12, 73, 0, 56],
                             [21, 80, 1, 85],
                             [20, 56, 0, 81],
                             [15, 78, 1, 72],
                             [12, 55, 0, 77],
                             [10, 92, 1, 80]]).to(torch.device('cuda'))

y_train = torch.FloatTensor([[123], [185], [177], [160], [179], [164]]).to(torch.device('cuda'))

x_max = torch.max(x_train).item()
y_max = torch.max(y_train).item()

x_train = torch.log(x_max - x_train + 1.5)
y_train = torch.log(y_max - y_train + 1.5)

# print(x_train)
# print(y_train)

w = torch.randn((4, 1), requires_grad=True, device='cuda')
b = torch.randn(1, requires_grad=True, device='cuda')

epoch = 1000
optimizer = optim.SGD([w, b], lr=1e-2)


for e in range(epoch):
    hypothesis = x_train.matmul(w) + b
    loss = torch.mean((y_train - hypothesis) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 10 == 0:
        print('{}/{} hypothesis: {}, loss: {}'.format(e, epoch, hypothesis.squeeze().detach(), round(loss.item(), 4)))