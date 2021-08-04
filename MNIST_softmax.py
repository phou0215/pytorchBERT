import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

from torch.utils.data import DataLoader



class MNISTModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True)
        )

    def forward(self, x):
        return self.model(x)


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print('Training device : ', device)

random.seed(2323)
torch.manual_seed(2323)
if device == 'cuda':
    torch.cuda.manual_seed_all(2323)
epoch = 15
batch_size = 100

mnist_train = datasets.MNIST(root='MNIST_data/',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)
mnist_test = datasets.MNIST(root='MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

dataloader = DataLoader(dataset=mnist_train,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

testloader = DataLoader( dataset=mnist_test,
                         batch_size=batch_size,
                         shuffle=True)

model = MNISTModel(28*28, 10).to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss().to(device)

model.train()
for e in range(epoch):
    avg_loss = 0
    total_batch = len(dataloader)
    for idx, samples in enumerate(dataloader):
        x_data, y_data = samples
        x_data = x_data.view(-1, 28*28).to(device)
        y_data = y_data.to(device)
        hypothesis = model(x_data)
        loss = criterion(hypothesis, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss/total_batch

    print('Epoch {}/{} avg_loss:{:.4f}'.format(e+1, epoch, avg_loss))

model.eval()
with torch.no_grad():
    for idx, sample in enumerate(testloader):
            x_data, y_data = samples
            x_data = x_data.view(-1, 28*28).to(device)
            y_data = y_data.to(device)
            prediction = model(x_data)
            correct_reduce = torch.argmax(prediction, dim=1) == y_data
            accuracy = correct_reduce.float().mean()
            print('Accuracy : {}'.format(accuracy.item()))

         # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()