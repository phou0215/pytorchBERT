import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


torch.manual_seed(1)

class LogisticModel(nn.Module):

    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.logistic = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):

        return self.logistic(x)

model = LogisticModel(input_dim=2, output_dim=1).to(torch.device('cuda'))

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data).to(torch.device('cuda'))
y_train = torch.FloatTensor(y_data).to(torch.device('cuda'))
epoch = 1000

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, shuffle=True, batch_size=2)
optimizer = optim.SGD(model.parameters(), lr=1e-6)

for e in range(epoch):
    for idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        hypothesis = model(x_train)
        loss = F.binary_cross_entropy(hypothesis, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if e % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]).to(torch.device('cuda'))
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {}/{} loss: {:.4f} acc: {:.4f}%'.format(e, epoch, loss.item(), accuracy*100))

