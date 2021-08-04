import torch
import numpy as np

sentence_length = 10
word_dim = 4
hidden_size = 8

input_tensor = np.random.random((sentence_length, word_dim))
hidden_state_t = np.zeros((hidden_size,))

# 입력층에 대한 가중치 Tensor(8X4[가중치] * 4X1[입력 word 하나] = 8X1[hidden_size])
Wx = np.random.random((hidden_size, word_dim))
# 은닉층에(8X1)에 대한 가중치 Tensor(8X8[가중치] * 8X1[이전에 출력된 입력층의 결과] = 8X1[hidden_size])
Wh = np.random.random((hidden_size, hidden_size))
# bias 편향 값(8X1)
b = np.random.random((hidden_size,))

total_hidden_state = []

for input_t in input_tensor:
    out_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_state.append(out_t)
    print(np.shape(total_hidden_state))
    hidden_state_t = out_t

total_hidden_state = np.stack(total_hidden_state, axis = 0)
print(total_hidden_state)