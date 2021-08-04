import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader


class SNPModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 10, bias=True),
            nn.Sigmoid(),
            nn.Linear(10, 10, bias=True),
            nn.Sigmoid(),
            nn.Linear(10, 10, bias=True),
            nn.Sigmoid(),
            nn.Linear(10, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print('Training device : ', device)

torch.manual_seed(7)
if device == 'cuda':
    torch.cuda.manual_seed_all(7)
epoch = 10000
batch_size = 4

x_data = torch.FloatTensor([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]]).to(device)
y_data = torch.FloatTensor([[0],
                           [1],
                           [1],
                           [0]]).to(device)

dataset = TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

model = SNPModel(2, 1).to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)

model.train()
for e in range(epoch):
    avg_loss = 0.0
    total_batch_size = len(dataloader)
    for x_train, y_train in dataloader:
        hypothesis = model(x_train)
        loss = criterion(hypothesis, y_train)
        avg_loss += loss / total_batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss/total_batch_size
    if e % 100 == 0:
        print('Epoch {}/{} avg_loss: {}'.format(e, epoch, avg_loss))

# for e in range(epoch):
#     hypothesis = model(x_data)
#     loss = criterion(hypothesis, y_data)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if e % 100 == 0:
#         print('Epoch {}/{} loss: {}'.format(e, epoch, loss))

with torch.no_grad():
    hypothesis = model(x_data)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == y_data).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', y_data.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())
