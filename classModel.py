import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from linearModel import LinearRegressionModel as lm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


torch.manual_seed(1)
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]]).to(torch.device('cuda'))

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]]).to(torch.device('cuda'))

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, shuffle=True, batch_size=2)

model = lm(input_dim=3, output_dim=1).to(torch.device('cuda'))
optimizer = optim.SGD(model.parameters(), lr=1e-6, )
epoch = 1000
model.train()

for e in range(epoch):

    for idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        hypothesis = model(x_train)
        loss = F.mse_loss(hypothesis, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch {}/{} Batch {}/{} Cost: {:.4f}'.format(
            e+1, epoch, idx+1, len(dataloader),
            loss.item()))

x_pre = torch.FloatTensor([[73, 80, 75]]).to(torch.device('cuda'))
y_pre = model(x_pre)
print("예측값: {}".format(y_pre.item()))