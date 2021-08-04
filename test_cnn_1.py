import torch
import torch.optim as optim
import torch.nn as nn

# # 1번 레이어 : 합성곱층(Convolutional layer)
# 합성곱(in_channel = 1, out_channel = 32, kernel_size=3, stride=1, padding=1) + 활성화 함수 ReLU
# 맥스풀링(kernel_size=2, stride=2))
#
# # 2번 레이어 : 합성곱층(Convolutional layer)
# 합성곱(in_channel = 32, out_channel = 64, kernel_size=3, stride=1, padding=1) + 활성화 함수 ReLU
# 맥스풀링(kernel_size=2, stride=2))
#
# # 3번 레이어 : 전결합층(Fully-Connected layer)
# 특성맵을 펼친다. # batch_size × 7 × 7 × 64 → batch_size × 3136
# 전결합층(뉴런 10개) + 활성화 함수 Softmax

# 배치 크기 × 채널 × 높이(height) × 너비(widht)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)
# 1 layer
conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
# 2 layer
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# maxpooling
pool = nn.MaxPool2d(kernel_size=2, stride=2)

out = conv1(inputs)
print(out.shape)
out = pool(out)
print(out.shape)
out = conv2(out)
print(out.shape)
out = pool(out)
print(out.shape)

# change shape
out = out.view(out.size(0), -1)
print(out.shape)
