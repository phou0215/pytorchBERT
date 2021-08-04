import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.nn.init as init

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.keep_prob = 0.5
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 세번째층
        # ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        #  4번째층
        self.fc1 = nn.Linear(4*4*128, 625, bias=True)
        init.xavier_uniform_(self.fc1.weight)
        self.layer_4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob)
        )
        # 전층결합
        self.fc2 = nn.Linear(625, 10, bias=True)
        init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = out.view(out.size(0), -1)
        out = self.layer_4(out)
        out = self.fc2(out)
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(7777)

if device == 'cuda':
    torch.cuda.manual_seed_all(1)

learning_rate = 1e-2
epoch = 20
batch_size = 100

mnist_train = dataset.MNIST(root='MNIST_data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
mnist_test = dataset.MNIST(root='MNIST_data/',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

dataloader = DataLoader(mnist_train, shuffle=True, batch_size=batch_size, drop_last=True)
model = CNNModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)

for e in range(epoch):
    avg_loss = 0
    total_batch = len(dataloader)

    for idx, samples in enumerate(dataloader):
        x_data, y_data = samples
        hypothesis = model(x_data.to(device))
        loss = criterion(hypothesis, y_data.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss
    print('Epoch {}/{} avg_loss:{:.4f}'.format(e+1, epoch, avg_loss))

model.eval()
with torch.no_grad():
    x_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    y_test = mnist_test.targets.float().to(device)
    prediction = model(x_test)
    correct = torch.argmax(prediction, dim=1) == y_test
    accuracy = correct.float().mean()
    print('Accuracy : {}'.format(accuracy))
