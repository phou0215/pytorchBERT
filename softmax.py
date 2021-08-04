import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from linearModel import LinearRegressionModel as lm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

torch.manual_seed(1)

class SoftMaxModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]

y_train = [2, 2, 2, 1, 1, 1, 0, 0]

x_train = torch.FloatTensor(x_train).to(torch.device('cuda'))
y_train = torch.LongTensor(y_train).to(torch.device('cuda'))

model = SoftMaxModel(input_dim=4, output_dim=3).to(torch.device('cuda'))
optimizer = optim.SGD(model.parameters(), lr=1e-4)
epoch = 1000

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, shuffle=True, batch_size=2)

for e in range(epoch):
    for idx, samples in enumerate(dataloader):
        x_data, y_data = samples
        hypothesis = model(x_data)
        loss = F.cross_entropy(hypothesis, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.4f}'.format(
            e, epoch, loss.item()
        ))


