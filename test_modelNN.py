import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


torch.manual_seed(3)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr=1e-5)
epoch = 1000

for e in range(epoch):
    predict = model(x_train)
    loss = F.mse_loss(predict, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 10 == 0:
        print('{}/{} loss:{}'.format(e+1, epoch, loss.item()))

# test
x_value = torch.FloatTensor([[96, 98, 100]])
pre_value = model(x_value)
print(pre_value.item())
print(list(model.parameters()))